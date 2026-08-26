from __future__ import annotations

import io
from pathlib import Path
import stat
import zipfile

import pytest
from fastapi.testclient import TestClient

from backend.routes import robot
from backend.services.preview_renderer import (
    PreviewError,
    validate_model_asset_confinement,
)


def _archive(*members: tuple[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, body in members:
            zf.writestr(name, body)
    return payload.getvalue()


def _make_project(client: TestClient, name: str) -> str:
    response = client.post(
        "/projects", json={"name": name, "adapter": "gym_sb3"}
    )
    assert response.status_code == 201, response.text
    return response.json()["slug"]


def test_archive_rejects_member_count_before_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(robot, "MAX_ZIP_MEMBERS", 2)
    payload = _archive(
        ("a.stl", b"a"),
        ("b.stl", b"b"),
        ("c.stl", b"c"),
    )

    with pytest.raises(robot._ZipReject, match="members"):
        robot._extract_mesh_zip(payload, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_archive_rejects_aggregate_expansion_before_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(robot, "MAX_ZIP_EXPANDED_BYTES", 5)
    payload = _archive(("mesh.stl", b"123456"))

    with pytest.raises(robot._ZipReject, match="expands"):
        robot._extract_mesh_zip(payload, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_archive_rejects_casefold_duplicate_and_code(tmp_path: Path) -> None:
    duplicate = _archive(("mesh/A.stl", b"a"), ("MESH/a.STL", b"b"))
    with pytest.raises(robot._ZipReject, match="duplicate"):
        robot._extract_mesh_zip(duplicate, tmp_path)

    code = _archive(("postprocess.py", b"print('no')"))
    with pytest.raises(robot._ZipReject, match="not an admitted"):
        robot._extract_mesh_zip(code, tmp_path)


def test_archive_rejects_links_and_executable_members(tmp_path: Path) -> None:
    link_payload = io.BytesIO()
    with zipfile.ZipFile(link_payload, "w") as zf:
        link = zipfile.ZipInfo("body.stl")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(link, b"../../outside")
    with pytest.raises(robot._ZipReject, match="links"):
        robot._extract_mesh_zip(link_payload.getvalue(), tmp_path)

    executable_payload = io.BytesIO()
    with zipfile.ZipFile(executable_payload, "w") as zf:
        executable = zipfile.ZipInfo("body.stl")
        executable.create_system = 3
        executable.external_attr = (stat.S_IFREG | 0o755) << 16
        zf.writestr(executable, b"solid body\nendsolid body\n")
    with pytest.raises(robot._ZipReject, match="executable"):
        robot._extract_mesh_zip(executable_payload.getvalue(), tmp_path)


def test_archive_rejects_pathological_compression_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(robot, "MAX_ZIP_RATIO", 2)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("body.stl", b"0" * 4096)

    with pytest.raises(robot._ZipReject, match="compression ratio"):
        robot._extract_mesh_zip(payload.getvalue(), tmp_path)


def test_archive_accepts_data_only_assets(tmp_path: Path) -> None:
    payload = _archive(
        ("meshes/body.stl", b"solid body\nendsolid body\n"),
        ("textures/body.png", b"not decoded during admission"),
    )

    extracted = robot._extract_mesh_zip(payload, tmp_path)

    assert extracted == ["meshes/body.stl", "textures/body.png"]
    assert (tmp_path / "meshes" / "body.stl").is_file()


def test_archive_rejects_obj_secondary_loader(tmp_path: Path) -> None:
    payload = _archive(
        ("body.obj", b"mtllib ../../outside.mtl\nv 0 0 0\n"),
    )

    with pytest.raises(robot._ZipReject, match="material-library"):
        robot._extract_mesh_zip(payload, tmp_path)


@pytest.mark.parametrize(
    "reference",
    ["../outside.stl", "/tmp/outside.stl", "package://robot/body.stl"],
)
def test_model_confinement_rejects_external_asset_paths(
    tmp_path: Path, reference: str
) -> None:
    model = tmp_path / "robot.xml"
    model.write_text(
        f'<mujoco><asset><mesh file="{reference}"/></asset></mujoco>',
        encoding="utf-8",
    )

    with pytest.raises(PreviewError, match="asset reference") as caught:
        validate_model_asset_confinement(model, asset_root=tmp_path)

    assert caught.value.kind == "unsafe_model"


@pytest.mark.parametrize(
    "payload",
    [
        '<mujoco><include file="part.xml"/></mujoco>',
        '<mujoco><extension><plugin plugin="native"/></extension></mujoco>',
        '<!DOCTYPE mujoco [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        "<mujoco>&x;</mujoco>",
    ],
)
def test_model_confinement_rejects_loaders_and_entities(
    tmp_path: Path, payload: str
) -> None:
    model = tmp_path / "robot.xml"
    model.write_text(payload, encoding="utf-8")

    with pytest.raises(PreviewError) as caught:
        validate_model_asset_confinement(model, asset_root=tmp_path)

    assert caught.value.kind == "unsafe_model"


def test_model_confinement_accepts_child_asset_path(tmp_path: Path) -> None:
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><compiler meshdir="meshes"/>'
        '<asset><mesh file="body.stl"/></asset></mujoco>',
        encoding="utf-8",
    )

    validate_model_asset_confinement(model, asset_root=tmp_path)


def test_upload_size_rejection_uses_bounded_configurable_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(robot, "MAX_UPLOAD_BYTES", 16)
    slug = _make_project(client, "Bounded upload")

    response = client.post(
        f"/projects/{slug}/robot/urdf",
        files={"model_file": ("robot.xml", b"x" * 17, "application/xml")},
    )

    assert response.status_code == 413
    assert response.json()["type"] == "/problems/upload-too-large"


def test_upload_rejects_unconfined_model_before_mujoco(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client, "Unconfined model")
    response = client.post(
        f"/projects/{slug}/robot/urdf",
        files={
            "model_file": (
                "robot.xml",
                b'<mujoco><asset><mesh file="../secret.stl"/></asset></mujoco>',
                "application/xml",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["type"] == "/problems/unsafe-model"
    assert not (tmp_projects_root / slug / "uploads" / ".robot-incoming").exists()
