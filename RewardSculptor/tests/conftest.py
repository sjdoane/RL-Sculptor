"""Sculptor test-harness root conftest.

Provides:
  - `--regenerate-fixtures` CLI flag (session-scoped) for the GPU-gated
    smoke tests, so a subsequent `pytest -m gpu` run can reuse a cached
    checkpoint at `tests/fixtures/go1_smoke_checkpoint.pt` instead of
    paying the 5-minute mjlab train cost again.
  - Automatic skip for `@pytest.mark.gpu` tests when CUDA is not
    available. This keeps the default `uv run pytest tests/` fast and
    CI-friendly even on CPU-only boxes.
  - `mjlab_go1_checkpoint` session fixture that lazily trains on first
    use and caches the result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
GO1_CHECKPOINT_NAME = "go1_smoke_checkpoint.pt"


@pytest.fixture(autouse=True)
def _isolate_saved_root(tmp_path_factory, monkeypatch):
    """Mission-run auto-archives to `RS_SAVED_ROOT` on every stage/mission
    end (default `~/.local/share/reward-sculptor/saved/` — see
    `sculptor/archive.py::saved_root`). Without isolation, every test that
    drives a mission/stage to completion writes a real
    `pytest-*--mission--*` entry into the developer's actual saved-missions
    archive; ~95 such entries had accumulated there before this fixture.
    Point every test at a per-test temp dir instead. `RS_TRASH_ROOT` is
    read by the reward-sculptor-ui backend (not this library) but is
    isolated here too in case a future sculptor code path starts reading
    it. Tests that assert on env-resolution/override behavior set their
    own value via monkeypatch after this fixture runs and are unaffected."""
    monkeypatch.setenv(
        "RS_SAVED_ROOT",
        str(tmp_path_factory.mktemp("saved_root_isolated")),
    )
    monkeypatch.setenv(
        "RS_TRASH_ROOT",
        str(tmp_path_factory.mktemp("trash_root_isolated")),
    )


@pytest.fixture(autouse=True)
def _isolate_shared_kg(tmp_path_factory, monkeypatch):
    """2026-07-03: the KG default is now ALWAYS the user-wide shared DB
    (the cwd-legacy preference fragmented the graph and was removed).
    Without isolation, any test that constructs `SculptorKG()` bare —
    directly or via sculpt_run/diagnose — would read AND WRITE the
    developer's real graph at ~/.local/share/sculptor/kg/graph.db.
    Point every test at a per-test temp DB via the backend env alias;
    tests that assert on env-resolution behavior delete these vars
    themselves and are unaffected."""
    monkeypatch.setenv(
        "RS_KG_PATH",
        str(tmp_path_factory.mktemp("kg_isolated") / "graph.db"),
    )
    monkeypatch.delenv("SCULPTOR_KG_PATH", raising=False)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate-fixtures",
        action="store_true",
        default=False,
        help=(
            "Force re-creation of cached @gpu test fixtures "
            "(deletes tests/fixtures/go1_smoke_checkpoint.pt + re-trains). "
            "Without this flag, existing fixtures are reused."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip @gpu tests when CUDA is not present AND when the `gpu`
    marker was not explicitly selected via `-m gpu`."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available on this host")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda_ok = False
    for item in items:
        if "gpu" in item.keywords and not cuda_ok:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def regenerate_fixtures(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--regenerate-fixtures"))


@pytest.fixture(scope="session")
def mjlab_go1_checkpoint(regenerate_fixtures: bool) -> Path:
    """Session-scoped checkpoint fixture. First use on an RTX-class
    machine trains Go1 for 100 iters at num_envs=1024 via the mjlab
    runner subprocess; budget 5 min wall-clock. Subsequent use loads the
    cached file. `--regenerate-fixtures` wipes and retrains.

    Raises pytest.skip if CUDA is not available.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch not importable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FIXTURES_DIR / GO1_CHECKPOINT_NAME
    if regenerate_fixtures and cache_path.exists():
        cache_path.unlink()

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    # Cold cache — run the adapter.
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        num_envs=1024,
        device="cuda:0",
        max_iterations=100,
    )
    output_dir = FIXTURES_DIR / "_mjlab_go1_work"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = adapter.train(
        reward_module_path=None,  # default task reward — no sculpt injection
        output_dir=output_dir,
        steps=100,
        seed=1,
    )
    produced = result.checkpoint_path
    if not produced.is_file():
        raise RuntimeError(
            f"mjlab runner did not produce checkpoint at {produced}; "
            f"stdout/stderr captured by adapter.train()."
        )
    # Cache it.
    cache_path.write_bytes(produced.read_bytes())
    return cache_path
