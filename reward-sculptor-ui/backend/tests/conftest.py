"""Pytest fixtures.

Every test runs with `$RS_PROJECTS_ROOT` pointed at a unique tmp dir
(R7: smoke-test never adopts OneDrive-resident dirs). Cloud-sync
override is explicitly disabled so the guard's default path is
exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_network_adversarial(monkeypatch: pytest.MonkeyPatch):
    """§Metric-quality laws (LAW 9): the L3 adversarial gaming-archetype gate is
    DEFAULT ON in production (Sam's call) but makes a metric-blind LLM author call
    — at the novel-task grant AND the AUDIT-ONLY spec_* probe. Unit tests must not
    hit the network, so force the flag OFF by default; the dedicated audit test
    re-enables it with a mocked bridge. Read at module level, so monkeypatch the
    resolved constant, not the env var."""
    try:
        from backend.services import run_manager
        monkeypatch.setattr(run_manager, "_ADVERSARIAL_ENABLED", False)
    except Exception:  # noqa: BLE001 — tests that never import run_manager
        pass


@pytest.fixture
def tmp_projects_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "projects"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RS_PROJECTS_ROOT", str(root))
    monkeypatch.setenv("RS_ALLOW_CLOUD_SYNC", "false")
    # Redirect the user-wide shared KG into the per-test tmp dir so
    # the shared-default introduced in M7 Phase 1 doesn't pollute
    # ~/.local/share/sculptor/ during the backend test suite.
    monkeypatch.setenv("RS_KG_PATH", str(tmp_path / "shared_kg.db"))
    # §Ship-10: skip the sentence-transformers pre-warm during tests —
    # loading the ~90MB model on every TestClient spin-up makes the
    # suite unrunnable and masks which test is actually slow.
    monkeypatch.setenv("RS_SKIP_EMBEDDER_PREWARM", "1")
    return root


@pytest.fixture
def client(tmp_projects_root: Path):
    # Late imports — Settings() reads env at construction time.
    from backend.config import Settings
    from backend.main import create_app

    settings = Settings()
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c
