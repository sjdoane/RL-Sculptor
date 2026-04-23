"""Tests for the M7 Phase-1 shared KG (default path + bootstrap + stats endpoint).

Covers:
  - `shared_kg_db_path()` resolution precedence (RS_KG_PATH > SCULPTOR_KG_PATH > default).
  - `project_kg_db_path()` returns the shared path for new projects but
    preserves legacy per-project DBs in place.
  - Bootstrap hook in main.py copies a bundled pre-extracted DB into
    the shared path on cold start when absent.
  - `GET /system/kg/stats` returns aggregate counts, or a zero-filled
    response when the shared DB doesn't exist.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── unit tests over kg_store module ───────────────────────────────────
def test_shared_kg_path_env_var_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RS_KG_PATH overrides the default resolution."""
    from backend.services import kg_store

    target = tmp_path / "override.db"
    monkeypatch.setenv("RS_KG_PATH", str(target))
    monkeypatch.delenv("SCULPTOR_KG_PATH", raising=False)

    assert kg_store.shared_kg_db_path() == target.resolve()


def test_shared_kg_path_sculptor_env_var_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCULPTOR_KG_PATH works as a legacy alias when RS_KG_PATH is unset."""
    from backend.services import kg_store

    target = tmp_path / "sculptor_override.db"
    monkeypatch.delenv("RS_KG_PATH", raising=False)
    monkeypatch.setenv("SCULPTOR_KG_PATH", str(target))

    assert kg_store.shared_kg_db_path() == target.resolve()


def test_shared_kg_path_default_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without either env var, returns ~/.local/share/sculptor/kg/graph.db."""
    from backend.services import kg_store

    monkeypatch.delenv("RS_KG_PATH", raising=False)
    monkeypatch.delenv("SCULPTOR_KG_PATH", raising=False)

    resolved = kg_store.shared_kg_db_path()
    expected = (Path.home() / ".local/share/sculptor/kg/graph.db").resolve()
    assert resolved == expected


def test_project_kg_db_path_routes_new_projects_to_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new project with no legacy kg/graph.db → project_kg_db_path
    returns the shared DB path."""
    from backend.services import kg_store

    override = tmp_path / "shared.db"
    monkeypatch.setenv("RS_KG_PATH", str(override))

    project_dir = tmp_path / "newproj"
    project_dir.mkdir()
    resolved = kg_store.project_kg_db_path(project_dir)
    assert resolved == override.resolve()


def test_project_kg_db_path_preserves_legacy_per_project_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When <project>/kg/graph.db already exists (pre-Phase-1 project),
    `project_kg_db_path` returns the legacy path — no silent migration."""
    from backend.services import kg_store

    # Env override still points at the shared path — the legacy check
    # must win over the env so existing projects aren't disrupted.
    override = tmp_path / "shared.db"
    monkeypatch.setenv("RS_KG_PATH", str(override))

    project_dir = tmp_path / "legacy_proj"
    (project_dir / "kg").mkdir(parents=True)
    legacy_db = project_dir / "kg" / "graph.db"
    legacy_db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)

    resolved = kg_store.project_kg_db_path(project_dir)
    assert resolved == legacy_db


# ── /system/kg/stats endpoint ────────────────────────────────────────
def test_system_kg_stats_returns_zeros_when_db_absent(
    client: TestClient,
) -> None:
    """Fresh tmp project root → shared KG doesn't exist yet → endpoint
    returns a zero-filled response, NOT an error."""
    r = client.get("/system/kg/stats")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["db_exists"] is False
    assert body["papers"] == 0
    assert body["techniques"] == 0
    assert body["edges"] == 0
    # db_path is always present so the UI can surface it verbatim.
    assert body["db_path"]


def test_system_kg_stats_reports_counts_when_db_present(
    client: TestClient, tmp_path: Path
) -> None:
    """Materialize a minimal shared KG with one Paper node and confirm
    the endpoint reports it."""
    from sculptor.kg.schema import Paper, make_paper_id
    from sculptor.kg.store import SculptorKG

    shared_path = Path(os.environ["RS_KG_PATH"])
    arxiv_id = "1707.06347"
    with SculptorKG(shared_path) as kg:
        kg.add_node(
            Paper(
                id=make_paper_id(arxiv_id),
                arxiv_id=arxiv_id,
                title="Proximal Policy Optimization Algorithms",
                authors=["Schulman", "et al."],
                year=2017,
                extracted=False,
            )
        )

    r = client.get("/system/kg/stats")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["db_exists"] is True
    assert body["papers"] == 1
    assert body["db_size_bytes"] > 0
    assert body["last_modified"] is not None


# ── bootstrap hook behaviour ─────────────────────────────────────────
def test_bootstrap_no_op_when_rs_kg_path_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """With RS_KG_PATH set, the startup hook must NOT copy the bundled
    DB — otherwise tests would silently seed their own tmp DB with 46
    papers, slowing the suite and leaking bundled content into fixtures.
    """
    # conftest's tmp_projects_root fixture already sets RS_KG_PATH, so
    # the `client` fixture exercises the skip path. Verify no DB was
    # written at the fixture path.
    monkeypatch.setenv("RS_KG_PATH", str(tmp_path / "shouldnt_exist.db"))
    monkeypatch.setenv("RS_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("RS_ALLOW_CLOUD_SYNC", "false")
    (tmp_path / "projects").mkdir()

    # Force a fresh import path so create_app's startup hook runs.
    for mod in list(sys.modules):
        if mod.startswith("backend."):
            del sys.modules[mod]
    from backend.config import Settings
    from backend.main import create_app

    settings = Settings()
    app = create_app(settings=settings)
    with TestClient(app):
        pass

    assert not (tmp_path / "shouldnt_exist.db").exists()


def test_bootstrap_copies_bundled_db_when_shared_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no env vars are set AND the shared DB is missing AND a
    bundled pre-extracted DB exists next to the sculptor package, the
    startup hook copies it into the user's shared KG path."""
    import sculptor

    monkeypatch.delenv("RS_KG_PATH", raising=False)
    monkeypatch.delenv("SCULPTOR_KG_PATH", raising=False)

    # Redirect Path.home() so we don't touch ~/.local/share/sculptor.
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # Write a bundled DB alongside the sculptor package.
    sculptor_root = Path(sculptor.__file__).resolve().parent.parent
    bundled = sculptor_root / "examples" / "kg_preextracted.db"
    bundled_existed = bundled.is_file()
    if not bundled_existed:
        # Create a tiny valid-sqlite file as the stand-in bundled DB.
        bundled.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(bundled))
        conn.execute("CREATE TABLE probe (id INTEGER)")
        conn.commit()
        conn.close()
    try:
        monkeypatch.setenv("RS_PROJECTS_ROOT", str(tmp_path / "projects"))
        monkeypatch.setenv("RS_ALLOW_CLOUD_SYNC", "false")
        (tmp_path / "projects").mkdir()

        for mod in list(sys.modules):
            if mod.startswith("backend."):
                del sys.modules[mod]
        from backend.config import Settings
        from backend.main import create_app

        settings = Settings()
        app = create_app(settings=settings)
        with TestClient(app):
            pass

        shared = fake_home / ".local/share/sculptor/kg/graph.db"
        assert shared.is_file(), "bootstrap should have copied bundled DB"
        assert shared.read_bytes() == bundled.read_bytes()
    finally:
        if not bundled_existed and bundled.is_file():
            bundled.unlink()
