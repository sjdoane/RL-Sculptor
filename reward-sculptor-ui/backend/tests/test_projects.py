"""Tests for project-lifecycle endpoints + guardrails.

Covers:
  - GET /health reports sculptor_ok=true when sculptor is importable.
  - POST /projects scaffolds a dir with expected files (metadata.json,
    config.toml, rewards/v0.py, kg/, uploads/).
  - Slug derivation + collision suffix.
  - GET /projects (list) and GET /projects/{slug} (detail).
  - DELETE /projects/{slug} removes the dir and 404s on repeat.
  - 404 on missing project.
  - Accepts both `name` and `display_name` in the create body.
  - R7: cloud-sync guard aborts on OneDrive-resident paths; override
    env var bypasses.
  - CRITICAL: backend does NOT modify any file under the sculptor
    source tree while handling a full create/delete cycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── health ─────────────────────────────────────────────────────────────
def test_health_reports_sculptor_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["sculptor_ok"] is True, body
    assert body["status"] == "ok"
    assert body["sculptor_error"] is None
    assert body["projects_root"]


# ── POST + GET ─────────────────────────────────────────────────────────
def test_create_then_get_project(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.post(
        "/projects",
        json={"name": "Test Quadruped", "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    slug = body["slug"]
    assert slug == "test-quadruped"
    assert body["display_name"] == "Test Quadruped"
    # "ready" once the user-wide shared KG exists (bootstrap creates it);
    # "configured" only when KG resolution finds nothing (legacy fix:
    # the per-project kg/graph.db is never created anymore and used to
    # pin every project at "configured" forever).
    assert body["status"] in ("draft", "configured", "ready")
    assert body["adapter_class"] == "sculptor.adapters.gym_sb3.GymSB3Adapter"

    # Filesystem: sculpt_init outputs + UI-only dirs.
    project_dir = tmp_projects_root / slug
    assert project_dir.is_dir()
    assert (project_dir / "config.toml").is_file()
    assert (project_dir / "rewards" / "v0.py").is_file()
    assert (project_dir / "rewards" / "current.py").is_file()
    assert (project_dir / "kg_seeds.yml").is_file()
    assert (project_dir / ".gitignore").is_file()
    assert (project_dir / "metadata.json").is_file()
    assert (project_dir / "kg").is_dir()
    assert (project_dir / "uploads").is_dir()
    assert (project_dir / "runs").is_dir()
    assert (project_dir / "reports").is_dir()

    # GET detail.
    r = client.get(f"/projects/{slug}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["slug"] == slug
    assert detail["project_dir"] == str(project_dir)


def test_create_accepts_display_name_field(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.post(
        "/projects",
        json={"display_name": "Formal Name", "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["slug"] == "formal-name"


def test_create_persists_form_fields_to_sidecar(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """iteration_budget + behavior_goal from the create form land in
    metadata.json for later use by the run-start flow."""
    import json

    r = client.post(
        "/projects",
        json={
            "name": "Form Fields",
            "adapter": "gym_sb3",
            "description": "a project made from the UI",
            "iteration_budget": 35,
            "behavior_goal": "jump 10cm without falling",
        },
    )
    assert r.status_code == 201, r.text
    slug = r.json()["slug"]

    meta = json.loads(
        (tmp_projects_root / slug / "metadata.json").read_text("utf-8")
    )
    assert meta["iteration_budget"] == 35
    assert meta["behavior_goal"] == "jump 10cm without falling"
    assert meta["description"] == "a project made from the UI"


def test_create_rejects_bad_iteration_budget(client: TestClient) -> None:
    r = client.post(
        "/projects",
        json={"name": "Bad", "iteration_budget": 0},
    )
    assert r.status_code == 422
    r = client.post(
        "/projects",
        json={"name": "Bad2", "iteration_budget": 10_000},
    )
    assert r.status_code == 422


# ── list ───────────────────────────────────────────────────────────────
def test_list_projects(client: TestClient) -> None:
    client.post("/projects", json={"name": "Alpha"})
    client.post("/projects", json={"name": "Beta"})
    r = client.get("/projects")
    assert r.status_code == 200
    slugs = {p["slug"] for p in r.json()}
    assert {"alpha", "beta"} <= slugs


def test_list_empty(client: TestClient) -> None:
    r = client.get("/projects")
    assert r.status_code == 200
    assert r.json() == []


# ── delete ─────────────────────────────────────────────────────────────
def test_delete_project(client: TestClient, tmp_projects_root: Path) -> None:
    """Chunk A1: delete is a MOVE into the trash, not a hard delete —
    the project dir must be gone from projects_root but the tree
    (and its metadata.json) must still exist under `.trash/`."""
    r = client.post("/projects", json={"name": "ToDelete"})
    slug = r.json()["slug"]
    assert (tmp_projects_root / slug).is_dir()

    r = client.delete(f"/projects/{slug}")
    assert r.status_code == 204
    assert not (tmp_projects_root / slug).exists()

    # Gone from the live listing too.
    r = client.get("/projects")
    assert slug not in {p["slug"] for p in r.json()}

    # Recoverable: present in trash with the moved tree intact.
    trash_root = tmp_projects_root.parent / ".trash"
    entries = [d for d in trash_root.iterdir() if d.is_dir()]
    assert len(entries) == 1
    assert (entries[0] / "trash_meta.json").is_file()
    assert (entries[0] / slug / "metadata.json").is_file()

    # Repeat delete should 404 (already moved out of projects_root).
    r = client.delete(f"/projects/{slug}")
    assert r.status_code == 404
    assert r.json()["type"] == "/problems/not-found"


# ── 404 ────────────────────────────────────────────────────────────────
def test_get_missing_project_404(client: TestClient) -> None:
    r = client.get("/projects/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "/problems/not-found"
    assert body["status"] == 404


# ── slug collision ─────────────────────────────────────────────────────
def test_slug_collision_gets_suffix(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r1 = client.post("/projects", json={"name": "Same"})
    r2 = client.post("/projects", json={"name": "Same"})
    r3 = client.post("/projects", json={"name": "Same"})
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r3.status_code == 201
    assert r1.json()["slug"] == "same"
    assert r2.json()["slug"] == "same-2"
    assert r3.json()["slug"] == "same-3"
    for slug in ("same", "same-2", "same-3"):
        assert (tmp_projects_root / slug).is_dir()


# ── validation ─────────────────────────────────────────────────────────
def test_create_empty_name_422(client: TestClient) -> None:
    r = client.post("/projects", json={"name": "", "adapter": "gym_sb3"})
    assert r.status_code == 422


def test_create_extra_field_422(client: TestClient) -> None:
    r = client.post(
        "/projects",
        json={"name": "OK", "adapter": "gym_sb3", "bogus": 1},
    )
    assert r.status_code == 422


# ── cloud-sync guard (R7) ──────────────────────────────────────────────
def test_cloud_sync_guard_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolved path containing a cloud-sync segment must abort startup."""
    cloud_root = tmp_path / "OneDrive" / "reward-sculptor" / "projects"
    cloud_root.mkdir(parents=True)
    monkeypatch.setenv("RS_PROJECTS_ROOT", str(cloud_root))
    monkeypatch.delenv("RS_ALLOW_CLOUD_SYNC", raising=False)

    from backend.config import Settings
    from backend.main import create_app

    settings = Settings()
    with pytest.raises(SystemExit) as exc_info:
        create_app(settings=settings)
    assert exc_info.value.code == 2


def test_cloud_sync_override_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RS_ALLOW_CLOUD_SYNC=true downgrades the guard to a warning."""
    cloud_root = tmp_path / "OneDrive" / "reward-sculptor" / "projects"
    cloud_root.mkdir(parents=True)
    monkeypatch.setenv("RS_PROJECTS_ROOT", str(cloud_root))
    monkeypatch.setenv("RS_ALLOW_CLOUD_SYNC", "true")

    from backend.config import Settings
    from backend.main import create_app

    settings = Settings()
    # Should NOT raise.
    app = create_app(settings=settings)
    assert app is not None


def test_cloud_sync_guard_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guard hits Dropbox / Google Drive / iCloud too."""
    from backend.config import Settings, check_cloud_sync

    for segment in ("Dropbox", "Google Drive", "iCloud Drive", "Box"):
        p = tmp_path / segment / "rs" / "projects"
        p.mkdir(parents=True, exist_ok=True)
        hit = check_cloud_sync(p.resolve())
        assert hit is not None, f"segment {segment!r} should have matched"


# ── CRITICAL: no writes to sculptor source ─────────────────────────────
def test_no_writes_to_sculptor_source(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Snapshot mtimes under the sculptor package before + after a full
    project create/delete cycle. Any change is a constraint violation."""
    import sculptor

    sculptor_pkg = Path(sculptor.__file__).resolve().parent

    def snapshot() -> dict[Path, float]:
        out: dict[Path, float] = {}
        if not sculptor_pkg.is_dir():
            return out
        for p in sculptor_pkg.rglob("*"):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts:
                continue
            if p.suffix == ".pyc":
                continue
            try:
                out[p] = p.stat().st_mtime
            except (FileNotFoundError, PermissionError):
                continue
        return out

    before = snapshot()
    assert before, "sanity: sculptor source should contain files"

    r = client.post("/projects", json={"name": "Audit"})
    assert r.status_code == 201
    slug = r.json()["slug"]
    r = client.delete(f"/projects/{slug}")
    assert r.status_code == 204

    after = snapshot()
    changed = {p for p in before if p in after and before[p] != after[p]}
    new_files = set(after) - set(before)
    removed = set(before) - set(after)

    assert not changed, f"sculptor source files modified: {sorted(changed)}"
    assert not new_files, f"sculptor source files created: {sorted(new_files)}"
    assert not removed, f"sculptor source files removed: {sorted(removed)}"


# ── §Ship-8: per-project settings GET/PATCH ─────────────────────────────
def test_get_project_settings_returns_iteration_block(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    r = client.post("/projects", json={"name": "Settings"})
    assert r.status_code == 201
    slug = r.json()["slug"]
    r = client.get(f"/projects/{slug}/settings")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "iteration" in body
    it = body["iteration"]
    assert it["steps_per_iter"] == 50_000
    assert it["primary_metric"] == "mean_return"
    assert it["rollout_episodes"] == 6
    assert it["auto_adjust_physics"] is True


def test_patch_project_settings_upserts_new_and_existing_keys(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    r = client.post("/projects", json={"name": "PatchUpsert"})
    slug = r.json()["slug"]
    r = client.patch(
        f"/projects/{slug}/settings",
        json={
            "iteration": {
                "steps_per_iter": 1200,
                "max_episode_steps": 800,
                "playback_speed": 0.5,
                "auto_adjust_physics": False,
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()["iteration"]
    assert body["steps_per_iter"] == 1200
    assert body["max_episode_steps"] == 800
    assert body["playback_speed"] == 0.5
    assert body["auto_adjust_physics"] is False
    # rollout_episodes not touched → keeps scaffolded default.
    assert body["rollout_episodes"] == 6

    import tomllib
    with (tmp_projects_root / slug / "config.toml").open("rb") as f:
        on_disk = tomllib.load(f)["iteration"]
    assert on_disk["steps_per_iter"] == 1200
    assert on_disk["max_episode_steps"] == 800
    assert on_disk["playback_speed"] == 0.5
    assert on_disk["auto_adjust_physics"] is False


def test_patch_project_settings_partial_does_not_overwrite_unsent(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Sending just one field must preserve everything else on disk."""
    r = client.post("/projects", json={"name": "PatchPartial"})
    slug = r.json()["slug"]
    client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"steps_per_iter": 9999}},
    )
    r = client.get(f"/projects/{slug}/settings")
    body = r.json()["iteration"]
    assert body["steps_per_iter"] == 9999
    assert body["rollout_episodes"] == 6
    assert body["auto_adjust_physics"] is True
    assert body["primary_metric"] == "mean_return"


def test_patch_project_settings_rejects_extra_fields(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """`extra=forbid` on IterationSettings — unknown keys get 422."""
    r = client.post("/projects", json={"name": "PatchStrict"})
    slug = r.json()["slug"]
    r = client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"not_a_real_setting": 42}},
    )
    assert r.status_code == 422


def test_patch_project_settings_unknown_slug_404(client: TestClient) -> None:
    r = client.patch(
        "/projects/does-not-exist/settings",
        json={"iteration": {"steps_per_iter": 1000}},
    )
    assert r.status_code == 404


def test_patch_project_settings_toml_string_escaping(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """String values with TOML-special chars (quotes, backslashes) must
    round-trip safely."""
    r = client.post("/projects", json={"name": "StrEscape"})
    slug = r.json()["slug"]
    tricky = 'metric "with quotes" and \\ backslashes'
    r = client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"primary_metric": tricky}},
    )
    assert r.status_code == 200
    r = client.get(f"/projects/{slug}/settings")
    assert r.json()["iteration"]["primary_metric"] == tricky


def test_patch_project_settings_list_values_roundtrip(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """behavior_metrics is list[str] — must serialize as a TOML array."""
    r = client.post("/projects", json={"name": "ListVals"})
    slug = r.json()["slug"]
    r = client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"behavior_metrics": ["max_ep_len", "fall_rate"]}},
    )
    assert r.status_code == 200
    r = client.get(f"/projects/{slug}/settings")
    assert r.json()["iteration"]["behavior_metrics"] == ["max_ep_len", "fall_rate"]


def test_patch_project_settings_early_stop_flags_roundtrip(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """§Ship-9a: `early_stop_enabled` (bool) + `early_stop_patience`
    (int) PATCH → config.toml → GET must round-trip cleanly, including
    the bool→lowercase TOML serialization."""
    r = client.post("/projects", json={"name": "EarlyStopRT"})
    slug = r.json()["slug"]
    r = client.patch(
        f"/projects/{slug}/settings",
        json={
            "iteration": {
                "early_stop_enabled": False,
                "early_stop_patience": 10,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()["iteration"]
    assert body["early_stop_enabled"] is False
    assert body["early_stop_patience"] == 10

    # Verify TOML on disk uses lowercase `false`, not Python-style `False`.
    config_text = (tmp_projects_root / slug / "config.toml").read_text()
    assert "early_stop_enabled = false" in config_text
    # And re-parse survives via tomllib (no malformed byte leaked in).
    import tomllib
    with (tmp_projects_root / slug / "config.toml").open("rb") as f:
        parsed = tomllib.load(f)
    assert parsed["iteration"]["early_stop_enabled"] is False
    assert parsed["iteration"]["early_stop_patience"] == 10


# ── §Ship-8 hotfix regressions (from critique) ─────────────────────────

def test_patch_settings_does_not_clobber_other_sections(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Critical: upsert must only touch the [iteration] section.
    A `seed = N` under [adapter] (or elsewhere) must survive a
    PATCH that targets [iteration].seed."""
    r = client.post("/projects", json={"name": "NoClobber"})
    slug = r.json()["slug"]
    config = tmp_projects_root / slug / "config.toml"
    # Hand-inject a `seed = 777` into a NEW section that doesn't exist.
    config.write_text(
        config.read_text() + "\n[custom]\nseed = 777\n", encoding="utf-8"
    )
    # PATCH [iteration].seed — this would collide pre-hotfix.
    r = client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"seed": 42}},
    )
    assert r.status_code == 200, r.text
    import tomllib
    with config.open("rb") as f:
        parsed = tomllib.load(f)
    assert parsed.get("iteration", {}).get("seed") == 42
    # [custom].seed must survive untouched.
    assert parsed.get("custom", {}).get("seed") == 777


def test_patch_settings_creates_iteration_section_when_missing(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """If config.toml has no [iteration] table at all, PATCH must
    append a well-formed one (valid TOML on read-back)."""
    r = client.post("/projects", json={"name": "NoIterSection"})
    slug = r.json()["slug"]
    config = tmp_projects_root / slug / "config.toml"
    # Wipe the [iteration] block entirely — keep just [adapter].
    import tomllib
    with config.open("rb") as f:
        original = tomllib.load(f)
    assert "iteration" in original
    # Write a version without [iteration].
    new_text = (
        '[target]\nname = "NoIterSection"\n\n'
        '[adapter]\n'
        'class = "sculptor.adapters.gym_sb3.GymSB3Adapter"\n'
    )
    config.write_text(new_text, encoding="utf-8")

    r = client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"steps_per_iter": 5000}},
    )
    assert r.status_code == 200, r.text
    # File should still parse AND contain the new section.
    with config.open("rb") as f:
        parsed = tomllib.load(f)
    assert parsed["iteration"]["steps_per_iter"] == 5000
    assert parsed["adapter"]["class"].endswith("GymSB3Adapter")


def test_patch_settings_rejects_multi_line_string(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """TOML basic strings can't contain raw newlines — silently
    writing one would corrupt the file. Must 422."""
    r = client.post("/projects", json={"name": "MultiLine"})
    slug = r.json()["slug"]
    r = client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"primary_metric": "line1\nline2"}},
    )
    assert r.status_code == 422, r.text
    # File still parses.
    import tomllib
    with (tmp_projects_root / slug / "config.toml").open("rb") as f:
        tomllib.load(f)  # would raise if file got corrupted


def test_patch_settings_rejects_nan_and_inf(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """NaN / Inf in a float field must NOT end up in config.toml.
    FastAPI's pydantic layer rejects the payload with a validation
    error; we assert the critical invariant (file stays valid TOML)
    rather than a specific status code because error-body serialization
    of `inf` itself fails at the response layer (500-with-422-inside).
    """
    import json
    r = client.post("/projects", json={"name": "NaNInf"})
    slug = r.json()["slug"]
    config = tmp_projects_root / slug / "config.toml"
    original = config.read_text(encoding="utf-8")
    for bad in (float("inf"), float("-inf"), float("nan")):
        body = json.dumps(
            {"iteration": {"playback_speed": bad}}, allow_nan=True
        ).encode("utf-8")
        try:
            r = client.patch(
                f"/projects/{slug}/settings",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            # Either 422 (pydantic validation caught it) or 500
            # (response serialization failed on inf — still rejected).
            assert r.status_code in (400, 422, 500), (bad, r.status_code)
        except Exception:  # noqa: BLE001
            # Client-side JSON encoder may refuse — equally safe.
            pass
    # Invariant: file is unchanged, or at minimum still valid TOML.
    import tomllib
    with config.open("rb") as f:
        parsed = tomllib.load(f)
    # playback_speed should NOT be set to nan/inf on disk.
    ps = parsed.get("iteration", {}).get("playback_speed")
    if ps is not None:
        import math
        assert math.isfinite(float(ps)), f"file now has non-finite playback_speed={ps}"
    # Sanity: config.toml matches the original (no side-effect write).
    assert config.read_text(encoding="utf-8") == original


def test_toml_value_helper_rejects_unsafe_values() -> None:
    """Unit-test for `_toml_value` — catches the edge cases the route
    layer relies on turning into 422s."""
    import math
    from backend.routes.projects import _toml_value

    with pytest.raises(ValueError):
        _toml_value(None)
    with pytest.raises(ValueError):
        _toml_value(math.nan)
    with pytest.raises(ValueError):
        _toml_value(math.inf)
    with pytest.raises(ValueError):
        _toml_value("line1\nline2")
    with pytest.raises(ValueError):
        _toml_value({"not": "supported"})

    # Sanity: supported types round-trip cleanly.
    assert _toml_value(True) == "true"
    assert _toml_value(False) == "false"
    assert _toml_value(42) == "42"
    assert _toml_value(3.14) == "3.14"
    assert _toml_value("simple") == '"simple"'
    assert _toml_value(["a", "b"]) == '["a", "b"]'
    # Escape round-trip.
    assert _toml_value('q"x\\y') == '"q\\"x\\\\y"'


def test_patch_settings_empty_body_is_noop_not_422(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Empty {iteration: {}} must return 200 with the current state,
    not 422 — UI flow: open settings, revert to defaults, save."""
    r = client.post("/projects", json={"name": "EmptyPatch"})
    slug = r.json()["slug"]
    r = client.patch(
        f"/projects/{slug}/settings", json={"iteration": {}},
    )
    assert r.status_code == 200, r.text
    # Scaffold defaults still present.
    body = r.json()["iteration"]
    assert body["steps_per_iter"] == 50_000


def test_get_settings_surfaces_500_on_corrupt_toml(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Corrupt config.toml must return 500 — not hide the problem
    behind an empty form that would then be clobbered on save."""
    r = client.post("/projects", json={"name": "CorruptToml"})
    slug = r.json()["slug"]
    config = tmp_projects_root / slug / "config.toml"
    config.write_text("this is not [valid toml } = {", encoding="utf-8")
    r = client.get(f"/projects/{slug}/settings")
    assert r.status_code == 500
    assert "malformed" in r.json().get("detail", "").lower()


def test_patch_settings_no_trailing_newline(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """config.toml without a trailing newline must still round-trip."""
    r = client.post("/projects", json={"name": "NoTrailingNL"})
    slug = r.json()["slug"]
    config = tmp_projects_root / slug / "config.toml"
    text = config.read_text().rstrip("\n")  # strip trailing newline
    config.write_text(text, encoding="utf-8")
    r = client.patch(
        f"/projects/{slug}/settings",
        json={"iteration": {"steps_per_iter": 123}},
    )
    assert r.status_code == 200
    import tomllib
    with config.open("rb") as f:
        parsed = tomllib.load(f)
    assert parsed["iteration"]["steps_per_iter"] == 123


# ── §env generalization: read-only env-spec surface ─────────────────────
def test_env_spec_endpoint_inactive_then_active(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    import json as _json

    r = client.post("/projects", json={"name": "EnvSpec"})
    assert r.status_code == 201
    slug = r.json()["slug"]

    # No env spec yet → inactive, no versions.
    r = client.get(f"/projects/{slug}/env-spec")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["active"], body["current"], body["versions"]) == (
        False, None, [])
    # The editable set is advertised even with no spec on disk, so the UI
    # renders the same controls rather than hardcoding key names.
    assert "entropy_coef_scale" in body["editable_train_keys"]

    # Materialize a spec the way the loop does (v<N>.json + current copy).
    env_dir = tmp_projects_root / slug / "env"
    env_dir.mkdir()
    spec = {
        "env_spec_version": 1,
        "meta": {"version": "v0", "source": "generated"},
        "shared": {"episode_length_s": 10.0},
        "train": {"entropy_coef_scale": 2.0},
    }
    (env_dir / "v0.json").write_text(_json.dumps(spec))
    spec2 = {**spec, "meta": {"version": "v1", "source": "diagnoser"}}
    (env_dir / "v1.json").write_text(_json.dumps(spec2))
    (env_dir / "current.json").write_text(_json.dumps(spec2))

    r = client.get(f"/projects/{slug}/env-spec")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["active"] is True
    assert body["versions"] == ["v0", "v1"]
    assert body["current"]["meta"]["version"] == "v1"
    assert body["current"]["train"]["entropy_coef_scale"] == 2.0

    # Corrupt current.json degrades to inactive, versions still listed.
    (env_dir / "current.json").write_text("{not json")
    r = client.get(f"/projects/{slug}/env-spec")
    assert r.status_code == 200
    assert r.json()["active"] is False
    assert r.json()["versions"] == ["v0", "v1"]


def test_env_spec_endpoint_404_for_unknown_project(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    r = client.get("/projects/nope/env-spec")
    assert r.status_code == 404


# ── traversal-shaped slugs (regression: %2E%2E → 500) ──────────────────
def test_traversal_slug_is_404_not_500(
    client: TestClient, tmp_projects_root: Path
) -> None:
    import shutil

    # Slug ".." resolves to the PARENT of the projects root. Plant a
    # parseable metadata.json + config.toml there — before the slug guard
    # in ProjectStore.get this parsed fine, then blew up constructing
    # ProjectDetail(slug="..") with a pydantic ValidationError → HTTP 500.
    r = client.post("/projects", json={"name": "Victim"})
    assert r.status_code == 201
    slug = r.json()["slug"]
    parent = tmp_projects_root.parent
    shutil.copy(tmp_projects_root / slug / "metadata.json", parent)
    shutil.copy(tmp_projects_root / slug / "config.toml", parent)

    # %2E%2E must stay percent-encoded on the wire — httpx collapses a
    # literal "/../" client-side — so the route sees slug="..".
    for path in ("/projects/%2E%2E/env-spec", "/projects/%2E%2E"):
        r = client.get(path)
        assert r.status_code == 404, f"{path}: {r.status_code} {r.text}"
        assert r.json()["type"] == "/problems/not-found"


def test_store_get_rejects_malformed_slugs(tmp_path: Path) -> None:
    from backend.services.project_store import ProjectStore

    store = ProjectStore(tmp_path / "projects")
    for bad in ("..", ".", "a/b", "a\\b", "A", "-x", "x-", ""):
        assert store.get(bad) is None, bad


# ── PUT /projects/{slug}/env-spec/train ─────────────────────────────────
def _project_with_env_spec(client: TestClient, root: Path,
                           train: dict) -> str:
    import json as _json

    slug = client.post("/projects", json={"name": "EnvEdit"}).json()["slug"]
    env_dir = root / slug / "env"
    env_dir.mkdir()
    spec = {"env_spec_version": 1,
            "meta": {"version": "v0", "source": "generated"},
            "shared": {}, "train": train}
    (env_dir / "v0.json").write_text(_json.dumps(spec))
    (env_dir / "current.json").write_text(_json.dumps(spec))
    return slug


def test_editing_a_train_knob_writes_a_new_version_and_repoints_current(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """The knob that decides whether a run can succeed, reachable by hand.

    entropy_coef_scale at 3.0 triples PPO's entropy bonus; measured on
    platform-ascent-showcase the action-noise std climbed all run and
    mjlab's action_rate_l2 penalty overtook the task reward. Before this
    route the only way to change it was to wait for the diagnoser.
    """
    slug = _project_with_env_spec(
        client, tmp_projects_root, {"entropy_coef_scale": 3.0})

    r = client.put(f"/projects/{slug}/env-spec/train", json={"edits": [
        {"parameter": "entropy_coef_scale", "new_value": 1.0,
         "rationale": "std ratchet"}]})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] == ["entropy_coef_scale=1.0"]
    assert body["rejected"] == []
    assert body["new_version"] == "v1"
    assert body["current"]["train"]["entropy_coef_scale"] == 1.0
    # Persisted, and the GET agrees.
    got = client.get(f"/projects/{slug}/env-spec").json()
    assert got["versions"] == ["v0", "v1"]
    assert got["current"]["train"]["entropy_coef_scale"] == 1.0
    # A hand edit must not be recorded as the loop's own work.
    assert got["current"]["meta"]["source"] == "user"
    assert got["current"]["meta"]["parent"] == "v0"


def test_an_out_of_bounds_edit_is_422_with_the_reason_and_changes_nothing(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _project_with_env_spec(
        client, tmp_projects_root, {"entropy_coef_scale": 3.0})

    r = client.put(f"/projects/{slug}/env-spec/train", json={"edits": [
        {"parameter": "entropy_coef_scale", "new_value": 99.0}]})

    assert r.status_code == 422, r.text
    assert "entropy_coef_scale" in r.json()["detail"]
    got = client.get(f"/projects/{slug}/env-spec").json()
    assert got["versions"] == ["v0"], "a rejected edit writes no version"
    assert got["current"]["train"]["entropy_coef_scale"] == 3.0


def test_a_non_iterable_parameter_is_refused(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """The shared/eval section stays unreachable — train-only by design."""
    slug = _project_with_env_spec(
        client, tmp_projects_root, {"entropy_coef_scale": 3.0})

    r = client.put(f"/projects/{slug}/env-spec/train", json={"edits": [
        {"parameter": "episode_length_s", "new_value": 5.0}]})

    assert r.status_code == 422, r.text
    assert "episode_length_s" in r.json()["detail"]


def test_a_bad_edit_does_not_take_the_good_ones_down_with_it(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _project_with_env_spec(
        client, tmp_projects_root, {"entropy_coef_scale": 3.0})

    r = client.put(f"/projects/{slug}/env-spec/train", json={"edits": [
        {"parameter": "entropy_coef_scale", "new_value": 1.0},
        {"parameter": "min_base_height_termination_m", "new_value": 99.0}]})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] == ["entropy_coef_scale=1.0"]
    assert [p for p, _ in body["rejected"]] == [
        "min_base_height_termination_m"]
    assert body["current"]["train"]["entropy_coef_scale"] == 1.0
    assert "min_base_height_termination_m" not in body["current"]["train"]


def test_editing_train_on_a_project_with_no_env_spec_is_422(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = client.post("/projects", json={"name": "NoSpec"}).json()["slug"]
    r = client.put(f"/projects/{slug}/env-spec/train", json={"edits": [
        {"parameter": "entropy_coef_scale", "new_value": 1.0}]})
    assert r.status_code == 422, r.text


def test_editing_train_on_an_unknown_project_is_404(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    r = client.put("/projects/nope/env-spec/train", json={"edits": [
        {"parameter": "entropy_coef_scale", "new_value": 1.0}]})
    assert r.status_code == 404
