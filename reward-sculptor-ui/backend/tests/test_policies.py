"""Policy listing + export-bundle download routes.

Uses a tiny hand-built rsl_rl-format checkpoint (torch save of
{actor_state_dict: mlp.<i>.*}) so the full bundle path — including the
ONNX/TorchScript best-effort exports — runs in-process without GPU or
mjlab task registry (no task_id in config → export assumes elu and says
so in the manifest; that's fine for route-level tests).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")


def _make_project(client: TestClient) -> str:
    r = client.post(
        "/projects",
        json={"name": "Poly", "iteration_budget": 3, "behavior_goal": "hop"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _plant_iter(
    project_dir: Path, i: int, *, metric: float = 10.0,
    reward_version: str = "v0", fitness: float | None = None,
) -> None:
    it = project_dir / "runs" / f"iter_{i}"
    it.mkdir(parents=True, exist_ok=True)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.2.weight": torch.zeros(2, 8), "mlp.2.bias": torch.zeros(2),
        "distribution.std_param": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd, "iter": 5}, it / "checkpoint.pt")
    (it / "reward_spec.json").write_text(
        json.dumps({"version": reward_version}))
    (it / "metrics.json").write_text(
        json.dumps({"metrics": {"mean_return": metric}}))
    if fitness is not None:
        (it / "fitness.json").write_text(json.dumps({"fitness": fitness}))


def test_list_policies_empty(client: TestClient, tmp_projects_root: Path):
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/policies")
    assert r.status_code == 200
    assert r.json() == []


def test_list_policies_unknown_project(client: TestClient):
    r = client.get("/projects/nope/policies")
    assert r.status_code == 404


def test_list_policies_returns_disk_iters(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    pdir = tmp_projects_root / slug
    _plant_iter(pdir, 0, metric=5.0)
    _plant_iter(pdir, 2, metric=9.5, reward_version="v2", fitness=0.37)
    r = client.get(f"/projects/{slug}/policies")
    assert r.status_code == 200
    rows = r.json()
    assert [row["iter_index"] for row in rows] == [0, 2]
    assert rows[1]["primary_metric"] == pytest.approx(9.5)
    assert rows[1]["reward_version"] == "v2"
    assert rows[1]["fitness"] == pytest.approx(0.37)
    assert rows[0]["checkpoint"] == "checkpoint.pt"
    assert rows[0]["checkpoint_bytes"] > 0


def test_export_downloads_zip_bundle(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    pdir = tmp_projects_root / slug
    _plant_iter(pdir, 1)
    r = client.get(f"/projects/{slug}/policies/1/export")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert "iter1.zip" in r.headers.get("content-disposition", "")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert {"manifest.json", "checkpoint.pt", "DEPLOY.md",
            "reward/reward_spec.json", "config.toml"} <= names
    assert manifest["iter_index"] == 1
    assert manifest["checkpoint"]["format"] == "rsl_rl"
    # tiny MLP: dims inferred from the state dict
    assert manifest["network"]["obs_dim"] == 4
    assert manifest["network"]["action_dim"] == 2
    # bundle also persisted under the project for reuse
    assert (pdir / "exports").is_dir()
    assert any((pdir / "exports").glob("*.zip"))


def test_export_missing_iter_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    _plant_iter(tmp_projects_root / slug, 0)
    r = client.get(f"/projects/{slug}/policies/7/export")
    assert r.status_code == 404
    assert "iter 7" in r.json()["detail"]


def test_export_no_checkpoints_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/policies/0/export")
    assert r.status_code == 404
    assert "no exportable" in r.json()["detail"]


def test_export_negative_iter_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    _plant_iter(tmp_projects_root / slug, 0)
    r = client.get(f"/projects/{slug}/policies/-3/export")
    assert r.status_code == 404


def test_export_unknown_run_id_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    _plant_iter(tmp_projects_root / slug, 0)
    r = client.get(f"/projects/{slug}/policies?run_id=ghost")
    assert r.status_code == 404
    r = client.get(f"/projects/{slug}/policies/0/export?run_id=ghost")
    assert r.status_code == 404


def test_export_corrupt_checkpoint_still_ships_raw(
    client: TestClient, tmp_projects_root: Path,
):
    """A checkpoint torch can't read must still produce a bundle with the
    raw file + recipe (never a 500)."""
    slug = _make_project(client)
    pdir = tmp_projects_root / slug
    it = pdir / "runs" / "iter_0"
    it.mkdir(parents=True)
    (it / "checkpoint.pt").write_bytes(b"garbage")
    r = client.get(f"/projects/{slug}/policies/0/export")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "checkpoint.pt" in names
    assert "policy.onnx" not in names
    assert any("unreadable" in w for w in manifest["warnings"])
