"""Tests for §R1_BUILD_SPEC decision 11 — routes/references.py + the
StageSchema mirror of reference_clip_id/reference_tier/
reference_match_confidence.

A tiny fake library (1-2 clips, real clip.npz via
`sculptor.reference.save_clip`, provenance.json,
`sculptor.refs.library`-shaped index.jsonl, one preview.png) is built
on disk under a tmp `RS_REFERENCE_ROOT` — no network, no HuggingFace
download, no API key.

Covers: listing; deterministic search (use_llm off — the default);
detail; preview 404 when absent; path-traversal rejection; invalid
clip_id regex rejection; attach + detach on a mission fixture,
including the 409 guard when a mission-scoped job is live and the
pending-stage-with-no-training-dir case (§commit 8b0bfa3 precedent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── reference-library fixture helpers ──────────────────────────────────
def _write_clip_npz(clip_dir: Path, *, n_frames: int = 50, fps: float = 30.0) -> None:
    from sculptor.reference import save_clip

    root_pos_z = np.linspace(0.3, 0.8, n_frames).astype(np.float32)
    save_clip(clip_dir / "clip.npz", {"root_pos_z": root_pos_z, "fps": fps})


def _write_provenance(
    clip_dir: Path, *, clip_id: str, robot: str = "g1", text: str,
    labels: list[str], tier: str = "K", n_frames: int = 50,
    fps: float = 30.0, license_: str = "CC BY-NC-ND 4.0",
) -> dict:
    prov = {
        "schema": 1,
        "clip_id": clip_id,
        "robot": robot,
        "source": {"kind": "hf_dataset", "repo": "test/repo", "path": "g1/x.csv",
                    "url": "https://example.invalid/x.csv"},
        "license": license_,
        "attribution": "test dataset",
        "retarget": {"tool": "dataset-provided", "notes": ""},
        "tier": tier,
        "fps_source": fps,
        "parent_clip_id": None,
        "frame_range": None,
        "joint_mapping": {"identity": True},
        "content_sha256": "0" * 64,
        "labels": labels,
        "text": text,
        "qc": {"duration_s": n_frames / fps, "root_z_range": [0.3, 0.8], "checks": []},
        "ingested_at": "2026-07-09T00:00:00Z",
    }
    (clip_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def _write_preview(clip_dir: Path) -> None:
    # Minimal-but-real 1x1 PNG (avoids a mujoco/PIL dependency in tests).
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415478da6360606060000000050001a5f645400000000049"
        "454e44ae426082"
    )
    (clip_dir / "preview.png").write_bytes(png_bytes)


def _seed_library(
    root: Path, *, with_preview_for: set[str] | None = None,
) -> None:
    """Builds two clips: `fallandgetup1_subject1` (getup) and
    `walk1_subject2` (walk) — enough to exercise the acceptance-shaped
    query without downloading the real LAFAN1 dataset."""
    from sculptor.refs import library

    with_preview_for = with_preview_for or set()
    clips = [
        {
            "clip_id": "fallandgetup1_subject1",
            "text": "fall and get up (subject 1)",
            "labels": ["fall", "and", "get", "up", "subject1"],
            "tier": "K",
        },
        {
            "clip_id": "walk1_subject2",
            "text": "walk (subject 2)",
            "labels": ["walk", "subject2"],
            "tier": "K",
        },
    ]
    for c in clips:
        clip_dir = library.clip_dir("g1", c["clip_id"], root=root)
        clip_dir.mkdir(parents=True, exist_ok=True)
        _write_clip_npz(clip_dir)
        _write_provenance(
            clip_dir, clip_id=c["clip_id"], text=c["text"], labels=c["labels"],
            tier=c["tier"],
        )
        if c["clip_id"] in with_preview_for:
            _write_preview(clip_dir)

    library.rebuild_index(root=root)


def _seed_t1_clip(root: Path, *, with_preview: bool = False) -> None:
    """§Problem 2 (2026-07-11): one t1 clip, deliberately given the SAME
    text/labels as `fallandgetup1_subject1` (the g1 clip `_seed_library`
    writes) so a robot-filter bug — e.g. accidentally pooling every
    robot's rows, or hardcoding "g1" somewhere in the route layer —
    would be caught by a query that should hit only one of the two."""
    from sculptor.refs import library

    clip_id = "fallandgetup1_subject1_t1"
    clip_dir = library.clip_dir("t1", clip_id, root=root)
    clip_dir.mkdir(parents=True, exist_ok=True)
    _write_clip_npz(clip_dir)
    _write_provenance(
        clip_dir, clip_id=clip_id, robot="t1",
        text="fall and get up (subject 1)",
        labels=["fall", "and", "get", "up", "subject1"], tier="K")
    if with_preview:
        _write_preview(clip_dir)
    library.rebuild_index(root=root)


@pytest.fixture
def refs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    return root


# ── mission fixture helpers (mirrors test_mission_persistence.py) ─────
def _make_project(client: TestClient, name: str = "Refs Test") -> str:
    r = client.post("/projects", json={"name": name, "adapter": "gym_sb3"})
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _stage_dict(name: str, *, status: str = "pending") -> dict:
    return {
        "name": name,
        "goal_text": f"goal for {name}",
        "success_criterion": "metric > 0.5",
        "max_iterations": 4,
        "parent_stage": None,
        "reward_seed_prompt": f"seed for {name}",
        "kg_seed_papers": [],
        "status": status,
        "final_policy_path": None,
        "final_reward_path": None,
        "best_metric": None,
        "iterations_used": 0,
        "started_at": None,
        "finished_at": None,
        "redecomposition_attempts": 0,
    }


def _write_mission(project_dir: Path, mission_slug: str, stages: list[dict]) -> Path:
    md = project_dir / ".missions" / mission_slug
    md.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "goal": "test mission",
        "decomposition_model": "claude-opus-4-7",
        "decomposition_rationale": "test",
        "created_at": "2026-07-01T00:00:00+00:00",
        "current_stage_idx": 0,
        "stages": stages,
    }
    (md / "mission.json").write_text(json.dumps(payload))
    return md


# ── GET /references (listing + search) ─────────────────────────────────
def test_list_references_no_query_returns_slim_index(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    r = client.get("/references")
    assert r.status_code == 200, r.text
    rows = r.json()
    assert {row["clip_id"] for row in rows} == {
        "fallandgetup1_subject1", "walk1_subject2",
    }
    # Slim row shape — no full provenance leaking into the listing.
    assert set(rows[0].keys()) == {
        "clip_id", "robot", "text", "labels", "tier", "license",
        "n_frames", "fps", "duration_s", "root_z_range", "has_preview",
    }


def test_list_references_filters_by_robot(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    r = client.get("/references", params={"robot": "h1"})
    assert r.status_code == 200
    assert r.json() == []


def test_search_deterministic_ranks_getup_top_for_acceptance_query(
    client: TestClient, refs_root: Path,
) -> None:
    """§decision 7's acceptance query, exercised end-to-end through the
    HTTP layer with use_llm off (the default — no `llm=1` param)."""
    _seed_library(refs_root)
    r = client.get("/references", params={"q": "get up off the ground"})
    assert r.status_code == 200, r.text
    matches = r.json()
    assert matches, "expected at least one deterministic match"
    assert matches[0]["clip_id"] == "fallandgetup1_subject1"
    assert matches[0]["match_confidence"] is None
    assert matches[0]["rerank"] == "deterministic-only"


def test_search_respects_k(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root)
    r = client.get("/references", params={"q": "fall get up walk", "k": 1})
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_search_never_calls_llm_by_default(
    client: TestClient, refs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No network/API key required — the default `llm=0` path must
    never construct an anthropic client."""
    _seed_library(refs_root)

    import sculptor.refs.retrieve as retrieve_mod

    def _boom(*args, **kwargs):
        raise AssertionError("LLM rerank must not be invoked when llm=0")

    monkeypatch.setattr(retrieve_mod, "_rerank_with_llm", _boom)
    r = client.get("/references", params={"q": "get up off the ground"})
    assert r.status_code == 200, r.text


# ── robot symmetry (§Problem 2, 2026-07-11) ─────────────────────────────
def test_list_references_t1_clip_listed_symmetrically_with_g1(
    client: TestClient, refs_root: Path,
) -> None:
    """A t1 clip must be listable exactly like a g1 clip — `robot` is a
    real filter over whatever robots the index carries, not a g1-only
    special case (v1's "g1-only" API note is about not exposing a robot
    PATH segment, not about the query param only working for g1)."""
    _seed_library(refs_root)
    _seed_t1_clip(refs_root)

    r_g1 = client.get("/references", params={"robot": "g1"})
    assert r_g1.status_code == 200, r_g1.text
    assert {row["clip_id"] for row in r_g1.json()} == {
        "fallandgetup1_subject1", "walk1_subject2"}

    r_t1 = client.get("/references", params={"robot": "t1"})
    assert r_t1.status_code == 200, r_t1.text
    t1_rows = r_t1.json()
    assert {row["clip_id"] for row in t1_rows} == {"fallandgetup1_subject1_t1"}
    assert t1_rows[0]["robot"] == "t1"


def test_search_t1_clip_found_only_under_t1_robot(
    client: TestClient, refs_root: Path,
) -> None:
    """Same acceptance query as
    `test_search_deterministic_ranks_getup_top_for_acceptance_query`,
    scoped to `robot=t1` — the t1 clip (identical text to the g1 one)
    must rank top, and the g1 clip must not leak into t1's results."""
    _seed_library(refs_root)
    _seed_t1_clip(refs_root)

    r = client.get(
        "/references", params={"q": "get up off the ground", "robot": "t1"})
    assert r.status_code == 200, r.text
    matches = r.json()
    assert matches, "expected at least one deterministic match for robot=t1"
    assert matches[0]["clip_id"] == "fallandgetup1_subject1_t1"


# ── GET /references/{clip_id} ────────────────────────────────────────
def test_get_reference_detail(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root)
    r = client.get("/references/fallandgetup1_subject1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["index_row"]["clip_id"] == "fallandgetup1_subject1"
    assert body["provenance"]["clip_id"] == "fallandgetup1_subject1"
    assert body["provenance"]["license"] == "CC BY-NC-ND 4.0"


def test_get_reference_detail_404_unknown_clip(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    r = client.get("/references/no-such-clip")
    assert r.status_code == 404


def test_get_reference_detail_404_invalid_clip_id_regex(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    # Uppercase + traversal-shaped path segments both fail the
    # `^[a-z0-9][a-z0-9_-]{0,95}$` guard.
    for bad_id in ["Bad-ID", "..", "has space"]:
        r = client.get(f"/references/{bad_id}")
        assert r.status_code == 404, (bad_id, r.text)


# ── GET /references/{clip_id}/preview ────────────────────────────────
def test_preview_404_when_absent(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root)  # no previews written
    r = client.get("/references/fallandgetup1_subject1/preview")
    assert r.status_code == 404


def test_preview_200_when_present(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root, with_preview_for={"fallandgetup1_subject1"})
    r = client.get("/references/fallandgetup1_subject1/preview")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"


# ── GET /references/{clip_id}/file/clip.npz ──────────────────────────
def test_download_clip_file(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root)
    r = client.get("/references/fallandgetup1_subject1/file/clip.npz")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    assert len(r.content) > 0


def test_download_clip_file_404_unknown_clip(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    r = client.get("/references/no-such-clip/file/clip.npz")
    assert r.status_code == 404


def test_download_clip_file_traversal_rejected(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    # FastAPI/Starlette resolve `..` path segments before routing, so a
    # literal ".." clip_id 404s at the clip_id-regex stage (still a
    # rejection, just earlier in the pipeline than the resolve/
    # relative_to guard it would otherwise hit).
    r = client.get("/references/../../etc/passwd/file/clip.npz")
    assert r.status_code in (404, 307, 308)


def test_get_reference_detail_preview_and_download_resolve_t1_clip(
    client: TestClient, refs_root: Path,
) -> None:
    """The single-clip routes (detail/preview/file) carry no `robot`
    param at all — robot is resolved by looking the clip_id up in the
    index (§decision 11's design). A t1 clip_id must resolve through
    every one of them exactly like a g1 clip_id does."""
    _seed_library(refs_root)
    _seed_t1_clip(refs_root, with_preview=True)
    clip_id = "fallandgetup1_subject1_t1"

    r = client.get(f"/references/{clip_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["index_row"]["robot"] == "t1"
    assert body["provenance"]["robot"] == "t1"

    r2 = client.get(f"/references/{clip_id}/preview")
    assert r2.status_code == 200, r2.text
    assert r2.headers["content-type"] == "image/png"

    r3 = client.get(f"/references/{clip_id}/file/clip.npz")
    assert r3.status_code == 200, r3.text
    assert len(r3.content) > 0


# ── attach / detach ─────────────────────────────────────────────────
def test_attach_reference_sets_stage_fields(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root, with_preview_for={"fallandgetup1_subject1"})
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("torso_righting")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/torso_righting/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reference_clip_id"] == "fallandgetup1_subject1"
    assert body["reference_tier"] == "K"
    assert body["reference_match_confidence"] is None

    # Persisted to mission.json.
    mission_json = json.loads((project_dir / ".missions" / "m1" / "mission.json").read_text())
    stage = mission_json["stages"][0]
    assert stage["reference_clip_id"] == "fallandgetup1_subject1"
    assert stage["reference_tier"] == "K"
    assert stage["reference_match_confidence"] is None

    # And it flows through the mission GET (StageSchema mirror).
    r2 = client.get(f"/projects/{slug}/missions/m1")
    assert r2.status_code == 200, r2.text
    stage2 = r2.json()["stages"][0]
    assert stage2["reference_clip_id"] == "fallandgetup1_subject1"
    assert stage2["reference_tier"] == "K"


def test_attach_reference_t1_clip(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """Attaching a t1 clip_id to a stage works identically to a g1 clip
    — the attach endpoint doesn't take a `robot` param at all, so this
    is really testing that `_find_index_row` (the clip_id -> row lookup
    every mutating route depends on) isn't secretly g1-only."""
    _seed_library(refs_root)
    _seed_t1_clip(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": "fallandgetup1_subject1_t1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reference_clip_id"] == "fallandgetup1_subject1_t1"
    assert body["reference_tier"] == "K"

    mission_json = json.loads((project_dir / ".missions" / "m1" / "mission.json").read_text())
    stage = mission_json["stages"][0]
    assert stage["reference_clip_id"] == "fallandgetup1_subject1_t1"


def test_attach_reference_pending_stage_without_training_dir(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """§commit 8b0bfa3 precedent: a pending stage that has never
    trained has no stages/<stage>/ dir. Attach must validate against
    mission.json's stage list, not the training dir."""
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("torso_righting", status="pending")])
    assert not (project_dir / ".missions" / "m1" / "stages" / "torso_righting").exists()

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/torso_righting/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 200, r.text


def test_attach_reference_404_unknown_clip(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": "no-such-clip"},
    )
    assert r.status_code == 404


def test_attach_reference_404_unknown_stage(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/no-such-stage/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 404


def test_attach_reference_404_unknown_mission(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)

    r = client.post(
        f"/projects/{slug}/missions/no-such-mission/stages/a/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 404


def test_attach_reference_409_when_mission_job_active(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_execute",
        project_slug=slug,
        params={"mission_slug": "m1"},
    )

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 409, r.text


def test_attach_reference_409_when_stage_metric_regen_active(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """`active_mission_scoped_job` also covers mission_stage_run /
    mission_stage_metric_regen — the same non-atomic save_mission
    hazard applies to those as to decompose/execute."""
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_stage_metric_regen",
        project_slug=slug,
        params={"mission_slug": "m1", "stage_name": "a"},
    )

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 409, r.text


def test_detach_reference_clears_fields(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 200, r.text

    r = client.delete(f"/projects/{slug}/missions/m1/stages/a/reference")
    assert r.status_code == 204, r.text

    mission_json = json.loads((project_dir / ".missions" / "m1" / "mission.json").read_text())
    stage = mission_json["stages"][0]
    assert stage["reference_clip_id"] is None
    assert stage["reference_tier"] is None
    assert stage["reference_match_confidence"] is None


def test_detach_reference_409_when_mission_job_active(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 200, r.text

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_execute",
        project_slug=slug,
        params={"mission_slug": "m1"},
    )

    r = client.delete(f"/projects/{slug}/missions/m1/stages/a/reference")
    assert r.status_code == 409, r.text


def test_detach_reference_404_unknown_stage(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    r = client.delete(f"/projects/{slug}/missions/m1/stages/no-such-stage/reference")
    assert r.status_code == 404


# ── POST /references/compose ───────────────────────────────────────────
# Composition is how the system reaches a motion that exists in NO single
# clip: the goal's phases were each recorded, just never together. These
# pin the route's guards and the honesty of what it returns.
def _seed_composable(root: Path) -> None:
    """Two richer clips (joints + root pose) that can actually be composed."""
    import numpy as np

    from sculptor.reference import save_clip
    from sculptor.refs import library

    for idx, clip_id in enumerate(("src_alpha", "src_beta")):
        n, fps = 120, 60.0
        t = np.arange(n, dtype=np.float64) / fps
        jp = (0.05 * idx + 0.10 * np.sin(2 * np.pi * 0.5 * t)[:, None]
              + 0.01 * np.arange(4)[None, :])
        clip = {
            "fps": fps,
            "joint_names": [f"joint_{i}" for i in range(4)],
            "root_pos_z": 0.70 + 0.02 * np.sin(2 * np.pi * 0.5 * t),
            "root_pos_xy": np.stack([0.5 * t, np.zeros(n)], axis=1),
            "root_quat_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
            "joint_pos": jp,
        }
        d = library.clip_dir("g1", clip_id, root=root)
        save_clip(d / "clip.npz", clip)
        _write_provenance(d, clip_id=clip_id, text=f"source {clip_id}",
                          labels=["source"], n_frames=n, fps=fps)
    library.rebuild_index(root=root)


def test_compose_creates_a_novel_clip_from_two_sources(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_composable(refs_root)
    r = client.post("/references/compose", json={
        "clip_id": "novel-motion--g1",
        "robot": "g1",
        "text": "a motion no single clip contains",
        "segments": [
            {"clip_id": "src_alpha", "label": "approach"},
            {"clip_id": "src_beta", "label": "strike"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["clip_id"] == "novel-motion--g1"
    assert body["parent_clip_ids"] == ["src_alpha", "src_beta"]
    # Kinematic candidate, never presented as solved.
    assert body["tier"] == "K"
    assert body["certified"] is False
    assert "momentum is not conserved" in body["next_step"]
    # The seam measurements the caller needs to judge it are returned.
    assert body["qc"]["n_sources"] == 2
    assert len(body["qc"]["composition"]["seams"]) == 1

    # And it is immediately reachable through the normal library surface,
    # so the reference picker / stage attach need no special case.
    detail = client.get("/references/novel-motion--g1")
    assert detail.status_code == 200
    assert detail.json()["provenance"]["source"]["kind"] == "compose"


def test_compose_rejects_a_single_segment(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_composable(refs_root)
    r = client.post("/references/compose", json={
        "clip_id": "novel--g1", "robot": "g1",
        "segments": [{"clip_id": "src_alpha"}],
    })
    assert r.status_code == 400
    assert "at least 2" in r.json()["title"]


def test_compose_rejects_traversal_shaped_ids(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_composable(refs_root)
    r = client.post("/references/compose", json={
        "clip_id": "../../etc/passwd", "robot": "g1",
        "segments": [{"clip_id": "src_alpha"}, {"clip_id": "src_beta"}],
    })
    assert r.status_code == 400
    r2 = client.post("/references/compose", json={
        "clip_id": "novel--g1", "robot": "g1",
        "segments": [{"clip_id": "../../x"}, {"clip_id": "src_beta"}],
    })
    assert r2.status_code == 400


def test_compose_refuses_to_overwrite_an_existing_clip(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_composable(refs_root)
    r = client.post("/references/compose", json={
        "clip_id": "src_alpha", "robot": "g1",
        "segments": [{"clip_id": "src_alpha"}, {"clip_id": "src_beta"}],
    })
    assert r.status_code == 409


def test_compose_missing_source_is_a_400_not_a_500(
    client: TestClient, refs_root: Path,
) -> None:
    """A ComposeError is always a caller-fixable statement about the spans,
    so it must surface as a 400 carrying the real reason."""
    _seed_composable(refs_root)
    r = client.post("/references/compose", json={
        "clip_id": "novel--g1", "robot": "g1",
        "segments": [{"clip_id": "src_alpha"}, {"clip_id": "does_not_exist"}],
    })
    assert r.status_code == 400
    assert "not in the g1 library" in r.json()["detail"]


def test_compose_surfaces_the_seam_measurement_on_refusal(
    client: TestClient, refs_root: Path,
) -> None:
    """Spans that do not meet are refused in cheap kinematics rather than in
    an expensive tracking run — and the response says by how much."""
    import numpy as np

    from sculptor.reference import save_clip
    from sculptor.refs import library

    _seed_composable(refs_root)
    n, fps = 120, 60.0
    t = np.arange(n, dtype=np.float64) / fps
    far = {
        "fps": fps,
        "joint_names": [f"joint_{i}" for i in range(4)],
        "root_pos_z": 0.70 + 0.02 * np.sin(2 * np.pi * 0.5 * t),
        "root_pos_xy": np.stack([0.5 * t, np.zeros(n)], axis=1),
        "root_quat_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        "joint_pos": np.full((n, 4), 5.0),   # radians away from src_alpha
    }
    d = library.clip_dir("g1", "src_far", root=refs_root)
    save_clip(d / "clip.npz", far)
    _write_provenance(d, clip_id="src_far", text="far", labels=[],
                      n_frames=n, fps=fps)
    library.rebuild_index(root=refs_root)

    r = client.post("/references/compose", json={
        "clip_id": "novel--g1", "robot": "g1", "blend_s": 0.0,
        "segments": [{"clip_id": "src_alpha"}, {"clip_id": "src_far"}],
    })
    assert r.status_code == 400
    assert "seam discontinuity" in r.json()["detail"]
