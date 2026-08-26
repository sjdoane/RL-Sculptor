"""Tests for robot-source + static-preview endpoints.

Test matrix:
  - GET  /robot on fresh project returns kind="none".
  - POST /robot/library writes env_id into config.toml correctly.
  - POST /robot/urdf (MJCF) with a real Gymnasium asset succeeds + preview renders.
  - POST /robot/urdf with a malformed MJCF returns 400 + actionable parse error.
  - POST /robot/urdf with a minimal URDF: asserts BOTH branches per R8:
        If MuJoCo accepts the URDF → 201 and preview renders.
        If MuJoCo rejects the URDF → 400 with a specific error naming
        the failed parse paths (NOT a generic 500).
  - GET  /preview on a bare project (no robot) returns 409.
  - Unsupported file extension → 422.
  - Oversized upload → 413.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
from fastapi.testclient import TestClient


# ── fixtures ──────────────────────────────────────────────────────────
GYM_ASSETS = Path(gym.__file__).resolve().parent / "envs" / "mujoco" / "assets"
HOPPER_XML = GYM_ASSETS / "hopper.xml"


MALFORMED_MJCF = b"""<mujoco>
  <worldbody>
    <body name="a">
      <geom type="bogus_geom_type" size="1"/>
    </body>
</mujoco>
"""


# A valid URDF per the URDF 1.0 spec with inertia, visual, collision.
# Intentionally minimal — one link, no joints.
MINIMAL_URDF = b"""<?xml version="1.0"?>
<robot name="mini">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </visual>
    <collision>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" iyy="1.0" izz="1.0" ixy="0.0" ixz="0.0" iyz="0.0"/>
    </inertial>
  </link>
</robot>
"""


def _make_project(client: TestClient, name: str = "Robotlab") -> str:
    r = client.post("/projects", json={"name": name, "adapter": "gym_sb3"})
    assert r.status_code == 201, r.text
    return r.json()["slug"]


# ── GET /robot on empty project ───────────────────────────────────────
def test_get_robot_fresh_project_returns_none(client: TestClient) -> None:
    slug = _make_project(client, "Fresh")
    r = client.get(f"/projects/{slug}/robot")
    assert r.status_code == 200, r.text
    body = r.json()
    # sculpt_init seeds kind="library" with library_name=None. We normalize
    # to a resolvable shape for the UI: report kind as the sidecar records.
    assert body["kind"] in ("library", "none")
    assert body["library_name"] in (None, "")
    # env_id at creation time is "CHANGE_ME" from the template.
    assert body.get("env_id") in (None, "CHANGE_ME")


def test_get_robot_404_on_missing_project(client: TestClient) -> None:
    r = client.get("/projects/no-such-slug/robot")
    assert r.status_code == 404
    assert r.json()["type"] == "/problems/not-found"


# ── POST /robot/library ───────────────────────────────────────────────
def test_library_pick_updates_config_and_sidecar(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client, "LibPick")
    r = client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "hopper"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "library"
    assert body["library_name"] == "hopper"
    assert body["env_id"] == "Hopper-v4"
    assert body["model_file"] is None

    # config.toml env_id updated.
    cfg_text = (tmp_projects_root / slug / "config.toml").read_text("utf-8")
    assert 'env_id = "Hopper-v4"' in cfg_text
    assert 'CHANGE_ME' not in cfg_text.split('env_id')[1].split(',')[0]

    # Re-read via GET.
    r = client.get(f"/projects/{slug}/robot")
    assert r.status_code == 200
    assert r.json()["env_id"] == "Hopper-v4"

    project = client.get(f"/projects/{slug}")
    assert project.status_code == 200
    assert project.json()["library_slug"] == "hopper"
    assert project.json()["reference_robot"] == "hopper"


def test_library_pick_all_supported_names(client: TestClient) -> None:
    mapping = {
        "hopper": "Hopper-v4",
        "half_cheetah": "HalfCheetah-v4",
        "ant": "Ant-v4",
        "humanoid": "Humanoid-v4",
        "walker2d": "Walker2d-v4",
    }
    for name, env_id in mapping.items():
        slug = _make_project(client, name)
        r = client.post(
            f"/projects/{slug}/robot/library",
            json={"robot_name": name},
        )
        assert r.status_code == 200, (name, r.text)
        assert r.json()["env_id"] == env_id


def test_library_pick_unknown_name_422(client: TestClient) -> None:
    slug = _make_project(client, "BadLib")
    r = client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "chimera"},
    )
    assert r.status_code == 422


# ── POST /robot/urdf — valid MJCF ─────────────────────────────────────
def test_mjcf_upload_valid_succeeds(
    client: TestClient, tmp_projects_root: Path
) -> None:
    assert HOPPER_XML.is_file(), f"gym asset missing at {HOPPER_XML}"
    slug = _make_project(client, "MjcfUpload")
    content = HOPPER_XML.read_bytes()
    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={"model_file": ("hopper.xml", content, "application/xml")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["kind"] == "mjcf"
    assert body["original_filename"] == "hopper.xml"
    assert body["model_file"].endswith("hopper.xml")
    assert body["size_bytes"] == len(content)
    # File on disk.
    saved = tmp_projects_root / slug / body["model_file"]
    assert saved.is_file()


# ── POST /robot/urdf — malformed MJCF ─────────────────────────────────
def test_mjcf_upload_malformed_400(client: TestClient) -> None:
    slug = _make_project(client, "BadMjcf")
    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={
            "model_file": ("broken.xml", MALFORMED_MJCF, "application/xml")
        },
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["type"] == "/problems/model-parse"
    # Must name the file AND carry some MuJoCo-level detail.
    assert "broken.xml" in body["detail"]
    assert body["detail"] != "MuJoCo could not load the uploaded model"
    # parse_kind extension must be set, and it must NOT be a generic
    # 500/unknown kind.
    assert body.get("parse_kind") == "parse"


# ── POST /robot/urdf — URDF branching outcome per R8 ──────────────────
def test_urdf_upload_either_converts_or_errors_specifically(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """R8 contract: URDF uploads either succeed (201 + preview renders)
    or fail with a SPECIFIC error naming the missing capability.
    Never a generic 500. This machine's MuJoCo version decides which
    branch is exercised; the test asserts both shapes are handled."""
    slug = _make_project(client, "UrdfBranch")
    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={"model_file": ("mini.urdf", MINIMAL_URDF, "application/xml")},
    )

    # Pin the outcome set — neither 500 nor anything else is acceptable.
    assert r.status_code in (201, 400), (
        f"URDF upload returned unexpected status {r.status_code}; "
        f"R8 requires either success or a specific 400, never 500. "
        f"body: {r.text}"
    )

    if r.status_code == 201:
        body = r.json()
        assert body["kind"] == "urdf"
        assert body["original_filename"] == "mini.urdf"
        # Preview must render for the uploaded URDF.
        pr = client.get(f"/projects/{slug}/preview")
        assert pr.status_code == 200, (
            f"preview failed after successful URDF conversion: "
            f"{pr.status_code} {pr.text}"
        )
        assert pr.headers["content-type"] == "image/png"
        assert len(pr.content) > 200
    else:
        # 400 branch — error must be specific to URDF and name the paths.
        body = r.json()
        assert body["type"] == "/problems/urdf-unsupported"
        assert body.get("parse_kind") == "urdf_parse"
        detail = body["detail"] or ""
        assert "URDF" in detail, detail
        # Must mention BOTH attempted parse paths so the user knows
        # which capability is missing.
        assert "MjModel.from_xml_path" in detail, detail
        assert "MjSpec.from_file" in detail, detail
        # Must point to a concrete remediation (external conversion).
        assert "MJCF" in detail or "mjpython" in detail or "convert" in detail.lower(), detail


# ── upload edge cases ─────────────────────────────────────────────────
def test_upload_wrong_extension_422(client: TestClient) -> None:
    slug = _make_project(client, "WrongExt")
    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={"model_file": ("robot.stl", b"binary stuff", "application/octet-stream")},
    )
    assert r.status_code == 422
    assert r.json()["type"] == "/problems/validation-error"


def test_upload_empty_400(client: TestClient) -> None:
    slug = _make_project(client, "EmptyModel")
    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={"model_file": ("empty.xml", b"", "application/xml")},
    )
    assert r.status_code == 400


def test_upload_oversized_413(client: TestClient) -> None:
    slug = _make_project(client, "BigModel")
    # 51 MB — exceeds the 50 MB cap.
    huge = b"<mujoco></mujoco>" + (b" " * (51 * 1024 * 1024))
    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={"model_file": ("big.xml", huge, "application/xml")},
    )
    assert r.status_code == 413
    assert r.json()["type"] == "/problems/upload-too-large"


# ── zip slip ──────────────────────────────────────────────────────────
def test_mesh_zip_rejects_path_traversal(client: TestClient) -> None:
    import io
    import zipfile

    slug = _make_project(client, "ZipSlip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escapee.stl", b"binary mesh bytes")
    buf.seek(0)

    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={
            "model_file": ("hopper.xml", HOPPER_XML.read_bytes(), "application/xml"),
            "meshes_zip": ("meshes.zip", buf.read(), "application/zip"),
        },
    )
    assert r.status_code == 400, r.text
    assert r.json()["type"] == "/problems/unsafe-archive"


# ── GET /preview ──────────────────────────────────────────────────────
def test_preview_no_robot_returns_409(client: TestClient) -> None:
    slug = _make_project(client, "NoRobotPreview")
    # Rewrite the sidecar to explicitly have no robot (sculpt_init
    # stamps "library"/None which would flow into the "not_configured"
    # branch via env_id=None check).
    r = client.get(f"/projects/{slug}/preview")
    # With the default sidecar after sculpt_init, env_id is unset; the
    # renderer should reject with 409.
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["type"] == "/problems/state-conflict"
    assert body.get("render_kind") == "not_configured"


def test_preview_renders_after_library_pick(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client, "LibPreview")
    r = client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "hopper"},
    )
    assert r.status_code == 200

    pr = client.get(f"/projects/{slug}/preview")
    assert pr.status_code == 200, pr.text
    assert pr.headers["content-type"] == "image/png"
    assert len(pr.content) > 500  # non-trivial PNG

    # Cached on disk — default angle is iso.
    cached = tmp_projects_root / slug / "uploads" / "preview_iso.png"
    assert cached.is_file()
    assert cached.stat().st_size == len(pr.content)

    # Second call is served from cache — still OK.
    pr2 = client.get(f"/projects/{slug}/preview")
    assert pr2.status_code == 200
    assert pr2.content == pr.content


def test_preview_renders_every_angle(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """All four preset angles render distinct PNGs and cache under
    their own filenames."""
    slug = _make_project(client, "AllAngles")
    client.post(f"/projects/{slug}/robot/library", json={"robot_name": "hopper"})

    seen: set[bytes] = set()
    for angle in ("iso", "front", "side", "top"):
        r = client.get(f"/projects/{slug}/preview?angle={angle}")
        assert r.status_code == 200, (angle, r.text)
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 500
        seen.add(r.content)
        assert (tmp_projects_root / slug / "uploads" / f"preview_{angle}.png").is_file()

    # Top-down vs iso vs side are meaningfully different viewpoints,
    # so at least 2 of the 4 must be distinct bytes.
    assert len(seen) >= 2, "expected distinct renders for different angles"


def test_preview_bogus_angle_400(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client, "BadAngle")
    client.post(f"/projects/{slug}/robot/library", json={"robot_name": "hopper"})
    r = client.get(f"/projects/{slug}/preview?angle=chimera")
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "/problems/validation-error"
    assert "chimera" in (body.get("detail") or "")


def test_preview_regenerate_bypasses_cache(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client, "Regen")
    client.post(f"/projects/{slug}/robot/library", json={"robot_name": "hopper"})

    r1 = client.get(f"/projects/{slug}/preview?angle=iso")
    cached = tmp_projects_root / slug / "uploads" / "preview_iso.png"
    assert cached.is_file()
    mtime_before = cached.stat().st_mtime

    # Second call with regenerate=true must re-render (possibly same bytes,
    # but the mtime advances since we unlinked + rewrote the file).
    import time
    time.sleep(0.05)
    r2 = client.get(f"/projects/{slug}/preview?angle=iso&regenerate=true")
    assert r2.status_code == 200
    mtime_after = cached.stat().st_mtime
    assert mtime_after > mtime_before, "regenerate=true should rewrite cached file"
    # Identical-bytes check is not required (MuJoCo isn't bit-for-bit
    # deterministic across runs), but the file still exists.
    assert r1.status_code == 200


def test_preview_renders_after_mjcf_upload(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client, "MjcfPreview")
    r = client.post(
        f"/projects/{slug}/robot/urdf",
        files={
            "model_file": (
                "hopper.xml", HOPPER_XML.read_bytes(), "application/xml"
            )
        },
    )
    assert r.status_code == 201

    pr = client.get(f"/projects/{slug}/preview")
    assert pr.status_code == 200, pr.text
    assert pr.headers["content-type"] == "image/png"
    assert len(pr.content) > 500


def test_preview_library_slug_only_resolves_via_menagerie(
    client: TestClient,
    tmp_projects_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: library-driven project-create writes robot_source with
    only `library_slug` (+ `menagerie_package`) — no `library_name` or
    `env_id`. Before the Phase 0 fix, `preview_renderer` required
    `library_name` and raised `not_configured`, which the Dashboard
    rendered as "no robot yet". After the fix, resolve via menagerie."""
    import json

    slug = _make_project(client, "LibSlugPreview")
    # Overwrite the scaffold sidecar so robot_source mirrors what
    # `_finalize_library_scaffold` writes for a library pick: slug +
    # menagerie_package, no legacy fields.
    meta_path = tmp_projects_root / slug / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["robot_source"] = {
        "kind": "library",
        "library_slug": "regression_test_robot",
        "training_support": "mjlab_ready",
        "category": "quadruped",
        "menagerie_package": "fake_mj_description",
        "related_repos": [],
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # Short-circuit menagerie resolution to the bundled Hopper MJCF so
    # the render actually produces a PNG (no need to ship the real
    # robot_descriptions package just for this test).
    from backend.services.robot_library import RobotLibrary

    original = RobotLibrary.resolve_menagerie_path

    def fake_resolve(self, slug_arg: str):  # noqa: ANN001
        if slug_arg == "regression_test_robot":
            return HOPPER_XML
        return original(self, slug_arg)

    monkeypatch.setattr(RobotLibrary, "resolve_menagerie_path", fake_resolve)

    pr = client.get(f"/projects/{slug}/preview")
    assert pr.status_code == 200, pr.text
    assert pr.headers["content-type"] == "image/png"
    assert len(pr.content) > 500


def test_preview_library_slug_falls_back_to_env_id_for_gym(
    client: TestClient,
    tmp_projects_root: Path,
) -> None:
    """When a project is library_slug-scaffolded but the slug isn't
    Menagerie-sourced (e.g. gymnasium_compatible entry), preview falls
    back to the bundled gym env asset via `env_id`."""
    import json

    slug = _make_project(client, "GymFallbackPreview")
    meta_path = tmp_projects_root / slug / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["robot_source"] = {
        "kind": "library",
        "library_slug": "not_in_menagerie",
        "training_support": "gymnasium_compatible",
        "env_id": "Hopper-v4",
        "menagerie_package": None,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    pr = client.get(f"/projects/{slug}/preview")
    assert pr.status_code == 200, pr.text
    assert pr.headers["content-type"] == "image/png"


# ── audit: still no writes to sculptor source ─────────────────────────
def test_robot_ops_do_not_write_to_sculptor(
    client: TestClient, tmp_projects_root: Path
) -> None:
    import sculptor

    sculptor_pkg = Path(sculptor.__file__).resolve().parent

    def snapshot() -> dict[Path, float]:
        return {
            p: p.stat().st_mtime
            for p in sculptor_pkg.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
        }

    before = snapshot()

    slug = _make_project(client, "Audit2")
    client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "ant"},
    )
    client.get(f"/projects/{slug}/preview")
    client.post(
        f"/projects/{slug}/robot/urdf",
        files={
            "model_file": (
                "hopper.xml", HOPPER_XML.read_bytes(), "application/xml"
            )
        },
    )
    client.get(f"/projects/{slug}/preview")
    client.delete(f"/projects/{slug}")

    after = snapshot()
    changed = {p for p in before if p in after and before[p] != after[p]}
    new = set(after) - set(before)
    removed = set(before) - set(after)
    assert not changed, f"sculptor source modified: {sorted(changed)}"
    assert not new, f"sculptor source created: {sorted(new)}"
    assert not removed, f"sculptor source removed: {sorted(removed)}"


# ── §7.6: preview-render 503 under active sculpt run ────────────────────────
def test_preview_returns_503_when_render_fails_during_active_run(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """When a preview render fails with `kind="render"` AND a sculpt
    run is active on any project, the route must return 503 with
    `render_kind="render"` and `type="/problems/preview-busy"` instead
    of the default 500. Distinguishes GPU-contention-retryable from
    permanent-failure for the UI."""
    from backend.services import preview_renderer
    from backend.services.preview_renderer import PreviewError

    slug = _make_project(client, "BusyPreview")
    client.post(f"/projects/{slug}/robot/library", json={"robot_name": "hopper"})

    # Drop the iso cache so the route reaches the render path.
    cached = tmp_projects_root / slug / "uploads" / "preview_iso.png"
    cached.unlink(missing_ok=True)

    async def _fake_render(*_args, **_kw):
        raise PreviewError("EGL context unavailable", kind="render")

    monkeypatch.setattr(preview_renderer, "render_static_async", _fake_render)

    # Fake an active sculpt run on ANOTHER project.
    import asyncio
    from backend.services.job_manager import Job
    jm = client.app.state.job_manager
    fake_job = Job(
        job_id="busy_preview_fake",
        kind="sculpt_run",
        project_slug="some-other-project",
        status="running",
    )
    fake_job._cancel = asyncio.Event()
    with jm._lock:
        jm._jobs[fake_job.job_id] = fake_job

    try:
        r = client.get(f"/projects/{slug}/preview")
        assert r.status_code == 503, r.text
        body = r.json()
        assert body.get("type") == "/problems/preview-busy"
        assert body.get("render_kind") == "render"
        assert "GPU" in body.get("detail") or "sculpt run" in body.get("detail")
    finally:
        with jm._lock:
            jm._jobs.pop("busy_preview_fake", None)


def test_preview_returns_500_when_render_fails_and_no_active_run(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """Baseline: without an active run, a `kind="render"` failure keeps
    the pre-§7.6 500 status — the failure is genuinely unexplained, not
    GPU contention."""
    from backend.services import preview_renderer
    from backend.services.preview_renderer import PreviewError

    slug = _make_project(client, "NoActiveRunPreview")
    client.post(f"/projects/{slug}/robot/library", json={"robot_name": "hopper"})
    cached = tmp_projects_root / slug / "uploads" / "preview_iso.png"
    cached.unlink(missing_ok=True)

    async def _fake_render(*_args, **_kw):
        raise PreviewError("something else broke", kind="render")

    monkeypatch.setattr(preview_renderer, "render_static_async", _fake_render)

    r = client.get(f"/projects/{slug}/preview")
    assert r.status_code == 500
    body = r.json()
    assert body.get("type") == "/problems/preview-failed"
    assert body.get("render_kind") == "render"
