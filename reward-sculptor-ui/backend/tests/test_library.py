"""Tests for the robot library (M3).

Covers:
  - YAML load + schema validation (40-60 entries expected).
  - D-guard demotion when mjlab task_ids are missing from the registry.
  - GET /library/categories / robots / robots/{slug} shape.
  - Project creation via library_slug picks the right adapter + derives
    task_id / env_id / num_envs.
  - Auto-KG-seeding writes arxiv papers from references into kg_seeds.yml.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient


# ── library loader ──────────────────────────────────────────────────────────
def test_library_loads_without_errors() -> None:
    """M3 verification #1: robot_library.yml loads, 40-60 entries."""
    from backend.services.robot_library import load_library, reset_library_singleton

    reset_library_singleton()
    path = (
        Path(__file__).resolve().parents[1] / "data" / "robot_library.yml"
    )
    lib = load_library(path)
    n = len(lib.entries_by_slug)
    assert 40 <= n <= 70, f"expected 40-70 entries, got {n}"


def test_library_slugs_unique_and_well_formed() -> None:
    from backend.services.robot_library import _SLUG_RE, load_library

    path = (
        Path(__file__).resolve().parents[1] / "data" / "robot_library.yml"
    )
    lib = load_library(path)
    assert len(lib.entries_by_slug) > 0
    for slug in lib.entries_by_slug:
        assert _SLUG_RE.match(slug), f"malformed slug {slug!r}"


def test_library_categories_cover_mjlab_ready() -> None:
    """Every mjlab-ready entry lives in one of the enumerated categories."""
    from backend.services.robot_library import CATEGORIES, load_library

    path = (
        Path(__file__).resolve().parents[1] / "data" / "robot_library.yml"
    )
    # Bypass D-guard — test the YAML state, not the runtime-demoted state.
    lib = load_library(path)
    mjlab_entries = [
        e for e in lib.entries_by_slug.values()
        if e.training_support == "mjlab_ready"
    ]
    assert len(mjlab_entries) >= 5, (
        f"expected at least 5 mjlab_ready entries in YAML, got {len(mjlab_entries)}"
    )
    for e in mjlab_entries:
        assert e.category in CATEGORIES


def test_library_references_are_verified_urls() -> None:
    """M3 rule: references must be http(s) URLs. Loader enforces this."""
    from backend.services.robot_library import load_library

    path = (
        Path(__file__).resolve().parents[1] / "data" / "robot_library.yml"
    )
    lib = load_library(path)
    for entry in lib.entries_by_slug.values():
        for ref in entry.references:
            assert ref.url.startswith(("http://", "https://")), (
                f"{entry.slug}: non-http reference URL {ref.url!r}"
            )


def test_library_loader_rejects_duplicate_slugs(tmp_path: Path) -> None:
    from backend.services.robot_library import LibraryValidationError, load_library

    bad = tmp_path / "bad.yml"
    bad.write_text(
        "robots:\n"
        "  - slug: foo\n    display_name: Foo\n    category: Other\n"
        "    source: menagerie\n    training_support: preview_only\n"
        "    thumbnail_path: robots/foo.webp\n"
        "  - slug: foo\n    display_name: Foo Dup\n    category: Other\n"
        "    source: menagerie\n    training_support: preview_only\n"
        "    thumbnail_path: robots/foo.webp\n",
        encoding="utf-8",
    )
    with pytest.raises(LibraryValidationError, match="duplicate"):
        load_library(bad)


def test_library_loader_rejects_bad_url(tmp_path: Path) -> None:
    from backend.services.robot_library import LibraryValidationError, load_library

    bad = tmp_path / "bad.yml"
    bad.write_text(
        "robots:\n"
        "  - slug: foo\n    display_name: Foo\n    category: Other\n"
        "    source: menagerie\n    training_support: preview_only\n"
        "    thumbnail_path: robots/foo.webp\n"
        "    references:\n"
        "      - kind: paper\n        url: NOT-A-URL\n        citation: bad\n",
        encoding="utf-8",
    )
    with pytest.raises(LibraryValidationError, match="http"):
        load_library(bad)


# ── D-guard ─────────────────────────────────────────────────────────────────
def test_d_guard_demotes_when_tasks_not_registered() -> None:
    """If mjlab doesn't register a declared task_id, the entry gets
    demoted to preview_only with a note. Mock _fetch_mjlab_tasks so
    the test is fast + hermetic."""
    from backend.services.robot_library import load_library
    from backend.services import robot_library as lib_mod

    path = (
        Path(__file__).resolve().parents[1] / "data" / "robot_library.yml"
    )
    lib = load_library(path)
    # Ensure there's at least one mjlab_ready entry to demote.
    before = [
        e for e in lib.entries_by_slug.values()
        if e.training_support == "mjlab_ready"
    ]
    assert before, "expected mjlab_ready entries in YAML"

    # Mock the mjlab registry to an empty set — every mjlab_ready entry
    # should demote.
    with patch.object(lib_mod, "_fetch_mjlab_tasks", return_value=set()):
        lib._d_guard_applied = False
        lib._apply_d_guard()

    after_ready = [
        e for e in lib.entries_by_slug.values()
        if e.training_support == "mjlab_ready"
    ]
    assert len(after_ready) == 0, f"expected all demoted; still ready: {after_ready}"
    for e in before:
        entry = lib.entries_by_slug[e.slug]
        assert entry.training_support == "preview_only"
        assert entry.demote_note is not None
        assert entry.preconfigured_tasks == []


def test_d_guard_keeps_subset_of_valid_tasks() -> None:
    """If SOME tasks are registered, the entry stays mjlab_ready but
    unregistered tasks are filtered out of preconfigured_tasks."""
    from backend.services.robot_library import load_library
    from backend.services import robot_library as lib_mod

    path = (
        Path(__file__).resolve().parents[1] / "data" / "robot_library.yml"
    )
    lib = load_library(path)
    go1 = lib.entries_by_slug.get("unitree_go1")
    assert go1 is not None
    # Keep only one of Go1's tasks as "registered".
    keep = {go1.preconfigured_tasks[0].task_id}
    with patch.object(lib_mod, "_fetch_mjlab_tasks", return_value=keep):
        lib._d_guard_applied = False
        lib._apply_d_guard()
    go1 = lib.entries_by_slug["unitree_go1"]
    assert go1.training_support == "mjlab_ready"
    assert len(go1.preconfigured_tasks) == 1
    assert go1.preconfigured_tasks[0].task_id in keep


# ── endpoints ───────────────────────────────────────────────────────────────
def test_get_library_categories(client: TestClient) -> None:
    r = client.get("/library/categories")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert "Quadruped" in body
    assert "Humanoid" in body
    assert "Arm" in body


def test_get_library_robots_returns_all_entries(client: TestClient) -> None:
    r = client.get("/library/robots")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "robots" in body
    assert "total" in body
    assert body["total"] == len(body["robots"])
    assert body["total"] >= 40


def test_get_library_robots_filter_by_category(client: TestClient) -> None:
    r = client.get("/library/robots?category=Humanoid")
    assert r.status_code == 200
    body = r.json()
    for robot in body["robots"]:
        assert robot["category"] == "Humanoid"


def test_get_library_robots_search(client: TestClient) -> None:
    r = client.get("/library/robots?search=unitree")
    assert r.status_code == 200
    body = r.json()
    # Unitree G1, Go1, A1, H1, etc. all should match.
    assert body["total"] >= 2
    for robot in body["robots"]:
        hit = (
            "unitree" in robot["slug"].lower()
            or "unitree" in robot["display_name"].lower()
            or "unitree" in (robot["description"] or "").lower()
        )
        assert hit


def test_get_library_robot_detail(client: TestClient) -> None:
    r = client.get("/library/robots/unitree_g1")
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == "unitree_g1"
    assert body["category"] == "Humanoid"
    assert len(body["references"]) >= 3


def test_get_library_robot_404(client: TestClient) -> None:
    r = client.get("/library/robots/not-a-real-slug")
    assert r.status_code == 404
    assert r.headers.get("content-type", "").startswith("application/problem+json")


def test_library_thumbnail_404_when_missing(client: TestClient) -> None:
    """Thumbnails aren't generated in this test run; endpoint should
    return a structured 404 rather than crashing."""
    r = client.get("/library/robots/unitree_g1/thumbnail")
    # Either 200 (file present — in this repo, probably not) or 404.
    assert r.status_code in (200, 404)
    if r.status_code == 404:
        assert r.headers.get("content-type", "").startswith("application/problem+json")


# ── project creation via library_slug ──────────────────────────────────────
def test_create_project_with_library_slug_mjlab_ready(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """M3 verification #3: library_slug=unitree_g1 → adapter=mjlab,
    task_id populated, kg_seeds.yml has references."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mjlab project creation happy-path")

    from backend.services import robot_library as lib_mod
    # Mock D-guard so we don't pay mjlab import cost in a test.
    mock_tasks = {
        "Mjlab-Velocity-Flat-Unitree-G1",
        "Mjlab-Velocity-Rough-Unitree-G1",
        "Mjlab-Tracking-Flat-Unitree-G1",
        "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    }
    with patch.object(lib_mod, "_fetch_mjlab_tasks", return_value=mock_tasks):
        lib_mod.reset_library_singleton()
        r = client.post(
            "/projects",
            json={"name": "G1 From Library", "library_slug": "unitree_g1"},
        )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["adapter_class"] == "sculptor.adapters.mjlab.MjlabAdapter"
    assert body["adapter_config"]["task_id"] == "Mjlab-Velocity-Flat-Unitree-G1"
    assert body["library_slug"] == "unitree_g1"
    assert body["ready_to_train"] is True

    # kg_seeds.yml populated with arxiv paper references.
    slug = body["slug"]
    seeds = yaml.safe_load(
        (tmp_projects_root / slug / "kg_seeds.yml").read_text(encoding="utf-8")
    )
    papers = (seeds or {}).get("papers") or []
    arxiv_ids = [p.get("arxiv_id") for p in papers]
    # Unitree G1 has 3 arxiv papers in the library.
    assert len(arxiv_ids) >= 2
    for aid in arxiv_ids:
        assert aid and (aid[0].isdigit() or aid.startswith("v"))


def test_create_project_with_library_slug_hopper_gym(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """M3 verification #4: library_slug=hopper → gym_sb3, no regression."""
    from backend.services import robot_library as lib_mod
    lib_mod.reset_library_singleton()

    r = client.post(
        "/projects",
        json={"name": "Hopper From Library", "library_slug": "hopper"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["adapter_class"] == "sculptor.adapters.gym_sb3.GymSB3Adapter"
    assert body["library_slug"] == "hopper"
    assert body["ready_to_train"] is True


def test_create_project_with_library_slug_preview_only(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """M3 verification #5: library_slug=franka_fr3 → ready_to_train=false."""
    from backend.services import robot_library as lib_mod
    lib_mod.reset_library_singleton()

    r = client.post(
        "/projects",
        json={"name": "Franka FR3 Preview", "library_slug": "franka_fr3"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["adapter_class"] == "sculptor.adapters.gym_sb3.GymSB3Adapter"
    assert body["library_slug"] == "franka_fr3"
    assert body["ready_to_train"] is False


def test_create_project_rejects_unknown_library_slug(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.post(
        "/projects",
        json={"name": "Bogus", "library_slug": "not_a_real_slug"},
    )
    assert r.status_code == 404, r.text
    body = r.json()
    assert body["type"] == "/problems/library-slug-not-found"


def test_extract_arxiv_id_from_various_urls() -> None:
    from backend.routes.projects import _extract_arxiv_id

    assert _extract_arxiv_id("https://arxiv.org/abs/2406.08858") == "2406.08858"
    assert _extract_arxiv_id("https://arxiv.org/abs/2109.11978v2") == "2109.11978v2"
    assert _extract_arxiv_id("https://github.com/user/repo") is None
    assert _extract_arxiv_id("") is None
