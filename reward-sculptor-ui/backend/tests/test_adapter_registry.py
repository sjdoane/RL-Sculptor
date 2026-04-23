"""Tests for M5: adapter registry + coming-soon project creation."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── sculptor-side stub behavior ──────────────────────────────────────
def test_registry_contains_expected_adapters() -> None:
    from sculptor.adapters import ADAPTER_REGISTRY

    expected = {"gym_sb3", "mjlab", "isaac", "mjx", "rllib"}
    assert expected <= set(ADAPTER_REGISTRY), (
        f"registry missing adapters: {expected - set(ADAPTER_REGISTRY)}"
    )
    ready = {n for n, a in ADAPTER_REGISTRY.items() if a.status == "ready"}
    coming = {n for n, a in ADAPTER_REGISTRY.items() if a.status == "coming_soon"}
    assert ready == {"gym_sb3", "mjlab"}
    assert coming == {"isaac", "mjx", "rllib"}


def test_stub_adapters_raise_with_exact_message_format() -> None:
    """Each stub's NotImplementedError follows the exact format:
    `<Name> adapter not yet implemented. Adoption guide: <path>.
    Estimated effort: <effort>.`"""
    from sculptor.adapters.isaac_lab import IsaacLabAdapter
    from sculptor.adapters.mjx import MjxAdapter
    from sculptor.adapters.rllib import RllibAdapter

    cases = [
        (IsaacLabAdapter(), "Isaac Lab", "docs/adapters/isaac.md", "4-8 hours"),
        (MjxAdapter(), "Brax / MJX", "docs/adapters/mjx.md", "4-6 hours"),
        (RllibAdapter(), "Ray RLlib", "docs/adapters/rllib.md", "4-8 hours"),
    ]
    for adapter, name_frag, guide_frag, effort_frag in cases:
        with pytest.raises(NotImplementedError) as exc:
            adapter.train(
                reward_module_path=Path("/dev/null"),
                output_dir=Path("/tmp"),
                steps=1,
                seed=1,
            )
        msg = str(exc.value)
        assert name_frag in msg, f"missing name {name_frag!r} in {msg!r}"
        assert guide_frag in msg, f"missing guide {guide_frag!r} in {msg!r}"
        assert effort_frag in msg, f"missing effort {effort_frag!r} in {msg!r}"
        assert "not yet implemented" in msg


def test_stub_reward_contracts_are_valid() -> None:
    """Each stub's reward_contract() returns a valid RewardContract
    (needed so the UI can render the adapter in project lists without
    crashing on None-like values)."""
    from sculptor.adapters.base import RewardContract
    from sculptor.adapters.isaac_lab import IsaacLabAdapter
    from sculptor.adapters.mjx import MjxAdapter
    from sculptor.adapters.rllib import RllibAdapter

    for cls in (IsaacLabAdapter, MjxAdapter, RllibAdapter):
        c = cls().reward_contract()
        assert isinstance(c, RewardContract)
        assert isinstance(c.expected_info_keys, list)
        assert c.training_device in ("cpu", "gpu", "any")


# ── backend /library/adapters endpoint ──────────────────────────────
def test_get_library_adapters(client: TestClient) -> None:
    r = client.get("/library/adapters")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    names = {a["name"] for a in body}
    assert {"gym_sb3", "mjlab", "isaac", "mjx", "rllib"} <= names
    # Ready adapters come first.
    ready_idxs = [i for i, a in enumerate(body) if a["status"] == "ready"]
    coming_idxs = [i for i, a in enumerate(body) if a["status"] == "coming_soon"]
    assert max(ready_idxs) < min(coming_idxs)
    # Coming-soon adapters carry an adoption_guide_url + estimated_effort.
    for a in body:
        if a["status"] == "coming_soon":
            assert a["adoption_guide_url"].startswith("docs/adapters/")
            assert a["estimated_effort"]


# ── project creation with coming-soon adapter ────────────────────────
def test_create_project_with_coming_soon_adapter(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """M5 verification #3: project scaffolds, adapter_unavailable=true,
    ready_to_train=false, Training button disabled path."""
    r = client.post(
        "/projects",
        json={"name": "Isaac Test Project", "adapter": "isaac"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["adapter_unavailable"] is True
    assert body["ready_to_train"] is False
    assert body["adapter_class"] == "sculptor.adapters.isaac_lab.IsaacLabAdapter"

    project_dir = tmp_projects_root / body["slug"]
    toml_text = (project_dir / "config.toml").read_text()
    assert "sculptor.adapters.isaac_lab.IsaacLabAdapter" in toml_text


def test_create_project_gym_sb3_still_ready(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Regression: normal gym_sb3 project is ready_to_train=true."""
    r = client.post(
        "/projects",
        json={"name": "Classic Hopper", "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["adapter_unavailable"] is False
    assert body["ready_to_train"] is True


def test_adoption_guides_exist_on_disk() -> None:
    """Each ADAPTER_REGISTRY coming-soon entry points at an adoption
    guide that exists in the sculptor repo's docs/adapters/ dir."""
    from sculptor.adapters import ADAPTER_REGISTRY

    sculptor_root = Path(__file__).resolve().parents[3] / "RewardSculptor"
    for name, info in ADAPTER_REGISTRY.items():
        if info.status != "coming_soon":
            continue
        guide = sculptor_root / info.adoption_guide_url
        assert guide.is_file(), (
            f"{name}: adoption guide {info.adoption_guide_url} not "
            f"found at {guide}"
        )
        body = guide.read_text()
        # Sanity: each guide has the required sections.
        for required in ("Target version", "Install", "Reward injection", "References"):
            assert required in body, f"{name}: guide missing '{required}' section"
