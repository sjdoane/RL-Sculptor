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
