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

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── reference-library fixture helpers ──────────────────────────────────
def _write_clip_npz(clip_dir: Path, *, n_frames: int = 50, fps: float = 30.0) -> None:
    from sculptor.reference import save_clip

    root_pos_z = np.linspace(0.3, 0.8, n_frames).astype(np.float32)
    save_clip(clip_dir / "clip.npz", {
        "root_pos_z": root_pos_z,
        "fps": fps,
        "root_frame": "absolute",
    })


def _write_provenance(
    clip_dir: Path, *, clip_id: str, robot: str = "g1", text: str,
    labels: list[str], tier: str = "K", n_frames: int = 50,
    fps: float = 30.0, license_: str = "CC BY-NC-ND 4.0",
) -> dict:
    prov = {
        "schema": 2,
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
        "content_sha256": hashlib.sha256(
            (clip_dir / "clip.npz").read_bytes()
        ).hexdigest(),
        "source_content_sha256": "0" * 64,
        "labels": labels,
        "text": text,
        "qc": {
            "duration_s": (n_frames - 1) / fps,
            "root_z_range": [0.3, 0.8],
            "checks": [],
        },
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


def _seed_t1_clip(
    root: Path,
    *,
    with_preview: bool = False,
    clip_id: str = "fallandgetup1_subject1_t1",
) -> None:
    """§Problem 2 (2026-07-11): one t1 clip, deliberately given the SAME
    text/labels as `fallandgetup1_subject1` (the g1 clip `_seed_library`
    writes) so a robot-filter bug — e.g. accidentally pooling every
    robot's rows, or hardcoding "g1" somewhere in the route layer —
    would be caught by a query that should hit only one of the two."""
    from sculptor.refs import library

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


def _certify_clip(
    root: Path, robot: str, clip_id: str, *, project_dir: Path,
) -> None:
    """Write a minimal, internally consistent Tier-D test certificate."""
    from sculptor.refs import library
    from sculptor.policy_contract import build_project_policy_contract
    from sculptor.reference import load_clip, save_clip
    from sculptor.refs.track import (
        _score_tierd_rollout_artifact,
        bind_tierd_runtime_artifacts,
        build_tierd_execution_contract,
        build_tierd_reference_clock,
        downsample_phase_targets,
        update_provenance_tier_d,
    )
    from sculptor.runtime_inputs import environment_artifacts_for_phase

    clip_dir = library.clip_dir(robot, clip_id, root=root)
    clip = load_clip(clip_dir / library.CLIP_FILENAME)
    base_policy_contract = build_project_policy_contract(project_dir)
    ordered_joints = list(base_policy_contract["joints"]["ordered_names"])
    frame_count = len(np.asarray(clip["root_pos_z"]))
    clip["joint_names"] = ordered_joints
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count, dtype=np.float32)
    clip["joint_pos"] = np.repeat(
        (0.25 * np.sin(phase))[:, None], len(ordered_joints), axis=1,
    )
    clip["root_frame"] = "absolute"
    save_clip(clip_dir / library.CLIP_FILENAME, clip)
    clip_sha256 = library.content_sha256(
        (clip_dir / library.CLIP_FILENAME).read_bytes()
    )
    provenance = library.read_provenance(robot, clip_id, root=root)
    provenance["content_sha256"] = clip_sha256
    library.write_provenance(robot, clip_id, provenance, root=root)
    certified_clip = load_clip(clip_dir / library.CLIP_FILENAME)
    reference_clock = build_tierd_reference_clock(
        certified_clip, clip_id=clip_id, robot=robot,
    )
    policy_contract = build_project_policy_contract(
        project_dir, reference_clock=reference_clock,
    )
    execution_contract = build_tierd_execution_contract(
        donor_project=project_dir,
        certification_config_path=project_dir / "config.toml",
        clip_id=clip_id,
        robot=robot,
        clip=certified_clip,
        policy_contract=policy_contract,
        reference_clock=reference_clock,
    )
    reward_sha = "a" * 64
    checkpoint_sha = "b" * 64
    policy_contract_sha = execution_contract["donor"][
        "policy_contract_sha256"
    ]
    train_environment = environment_artifacts_for_phase(
        execution_contract["environment_artifacts"], "train",
    )
    execution_contract = bind_tierd_runtime_artifacts(
        execution_contract,
        requested_reward_module_sha256=reward_sha,
        train_receipts=[{
            "iteration": 1,
            "schema": "reward-sculptor-runner-artifacts-v2",
            "phase": "train",
            "reward_module_sha256": reward_sha,
            "requested_max_iterations": 2000,
            "requested_seed": 0,
            "requested_num_envs": 64,
            "seed_application": {
                "schema": "reward-sculptor-seed-application-v1",
                "applied_seed": 0,
                "python_random": True,
                "numpy_global": True,
                "torch_global": True,
                "env_cfg": True,
                "rl_cfg": True,
            },
            "environment_artifacts": train_environment,
            "env_spec_application": {
                "schema": "reward-sculptor-env-spec-application-v1",
                "phase": "train",
                "requested": [],
                "applied": [],
                "dead": [],
                "errors": [],
            },
            "input_checkpoint_requested_sha256": None,
            "input_checkpoint_loaded_sha256": None,
            "input_checkpoint_load_completed": False,
            "output_checkpoint_sha256": checkpoint_sha,
            "output_policy_contract_sha256": policy_contract_sha,
            "output_policy_contract_sidecar_sha256": "e" * 64,
        }],
        final_checkpoint_sha256=checkpoint_sha,
        requested_steps_per_iteration=2000,
        requested_seed=0,
        requested_num_envs=64,
    )
    dt = float(execution_contract["execution_boundary"]["timing"][
        "control_dt_s"
    ])
    duration_s = float(execution_contract["reference"][
        "playback_duration_s"
    ])
    n_steps = int(np.floor(duration_s / dt + 1e-12))
    sample_times = (np.arange(n_steps, dtype=np.float64) + 1.0) * dt
    phases = np.minimum(
        sample_times / duration_s, np.nextafter(1.0, 0.0),
    )
    indices = np.floor(
        phases * execution_contract["reference"]["phase_target_count"]
    ).astype(int)
    joint_targets = np.round(downsample_phase_targets(
        np.asarray(certified_clip["joint_pos"], dtype=np.float64), n=32,
    ), 5)
    root_targets = np.round(downsample_phase_targets(
        np.asarray(certified_clip["root_pos_z"], dtype=np.float64), n=32,
    ), 5)
    rollout_joint_pos = joint_targets[indices]
    rollout_root_pos = np.zeros((n_steps, 1, 3), dtype=np.float32)
    rollout_root_pos[:, 0, 2] = root_targets[indices]
    rollout_path = clip_dir / "tierD_rollout_candidate.npz"
    rollout_requirements = execution_contract["runtime_artifacts"][
        "rollout_requirements"
    ]
    metadata = {
        "schema": "reward-sculptor-trajectory-v1",
        "layout": ["time", "environment", "feature"],
        "ordered_joint_names": ordered_joints,
        "control_dt_s": dt,
        "root_link_pos_w_frame": "world",
        "first_episode_lane": 0,
        "valid_mask": {
            "key": "first_episode_valid_mask",
            "semantics": "true_prefix_before_first_done",
            "invalid_state": "frozen_last_valid_sample",
            "state_samples": "post_step_after_valid_transition",
        },
        "runtime_artifacts": {
            "schema": "reward-sculptor-runner-artifacts-v2",
            "phase": "rollout",
            "reward_module_sha256": reward_sha,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_load_completed": True,
            "environment_artifacts": rollout_requirements[
                "environment_artifacts"
            ],
            "requested_seed": 0,
            "applied_seed": 0,
            "seed_application": {
                "schema": "reward-sculptor-seed-application-v1",
                "applied_seed": 0,
                "python_random": True,
                "numpy_global": True,
                "torch_global": True,
                "env_cfg": True,
                "rl_cfg": True,
            },
            "requested_n_episodes": 1,
            "configured_n_episodes": 1,
            "requested_max_episode_steps": rollout_requirements[
                "requested_max_episode_steps"
            ],
            "configured_max_episode_steps": rollout_requirements[
                "requested_max_episode_steps"
            ],
            "requested_task_id": rollout_requirements["requested_task_id"],
            "configured_task_id": rollout_requirements["requested_task_id"],
            "configured_num_envs": 1,
            "completed_first_episodes": 0,
            "env_spec_application": {
                "schema": "reward-sculptor-env-spec-application-v1",
                "phase": "rollout",
                "requested": [],
                "applied": [],
                "dead": [],
                "errors": [],
            },
            "eval_reset_application": {
                "schema": "reward-sculptor-eval-reset-application-v1",
                "requested": [],
                "applied": [],
                "dead": [],
                "errors": [],
            },
        },
    }
    np.savez_compressed(
        rollout_path,
        joint_pos=rollout_joint_pos[:, None, :].astype(np.float32),
        root_link_pos_w=rollout_root_pos,
        first_episode_valid_mask=np.ones((n_steps, 1), dtype=bool),
        trajectory_contract_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"),
        )),
    )
    rollout_sha256 = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
    retained_rollout_path = clip_dir / f"tierD_rollout_{rollout_sha256}.npz"
    rollout_path.replace(retained_rollout_path)
    rollout_path = retained_rollout_path
    errors = _score_tierd_rollout_artifact(
        rollout_path,
        clip=certified_clip,
        execution_contract=execution_contract,
    )
    assert errors.feasible
    update_provenance_tier_d(
        robot=robot,
        clip_id=clip_id,
        errors=errors,
        iterations=1,
        rollout_path=rollout_path,
        execution_contract=execution_contract,
        root=root,
    )


@pytest.fixture
def refs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    return root


# ── mission fixture helpers (mirrors test_mission_persistence.py) ─────
def _make_project(client: TestClient, name: str = "Refs Test") -> str:
    r = client.post("/projects", json={"name": name, "adapter": "gym_sb3"})
    assert r.status_code == 201, r.text
    slug = r.json()["slug"]
    client.app.state.project_store.set_adapter_section(  # type: ignore[attr-defined]
        slug,
        "sculptor.adapters.mjlab.MjlabAdapter",
        {"task_id": "Mjlab-Velocity-Flat-Unitree-G1"},
    )
    client.app.state.project_store.write_robot_source(  # type: ignore[attr-defined]
        slug,
        {
            "kind": "library",
            "library_slug": "g1",
            "library_name": "g1",
            "training_support": "ready",
        },
    )
    return slug


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


def test_list_and_browse_downgrade_unverified_legacy_tier_d(
    client: TestClient,
    refs_root: Path,
) -> None:
    from sculptor.refs import library

    _seed_library(refs_root)
    clip_id = "fallandgetup1_subject1"
    provenance_path = (
        library.clip_dir("g1", clip_id, root=refs_root)
        / library.PROVENANCE_FILENAME
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["schema"] = 1
    provenance["tier"] = "D"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8",
    )
    library.rebuild_index(root=refs_root)

    listed = client.get("/references", params={"robot": "g1", "k": 100})
    assert listed.status_code == 200, listed.text
    row = next(item for item in listed.json() if item["clip_id"] == clip_id)
    assert row["tier"] == "K"
    assert row["claimed_tier"] == "D"

    browsed = client.get(
        "/references/browse", params={"robot": "g1", "limit": 100},
    )
    assert browsed.status_code == 200, browsed.text
    body = browsed.json()
    row = next(item for item in body["rows"] if item["clip_id"] == clip_id)
    assert row["tier"] == "K"
    assert body["facets"]["tiers"].get("D", 0) == 0

    filtered = client.get(
        "/references/browse",
        params={"robot": "g1", "tier": "D", "limit": 100},
    )
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["total"] == 0


# ── GET /references/{clip_id} ────────────────────────────────────────
def test_get_reference_detail(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root)
    r = client.get("/references/fallandgetup1_subject1", params={"robot": "g1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["index_row"]["clip_id"] == "fallandgetup1_subject1"
    assert body["provenance"]["clip_id"] == "fallandgetup1_subject1"
    assert body["provenance"]["license"] == "CC BY-NC-ND 4.0"
    assert body["dynamics_admission"]["admitted"] is False
    assert body["dynamics_admission"]["certificate_digest"] is None
    assert body["dynamics_admission"]["certification_scope"] is None
    assert body["artifact_identity"]["verified"] is True
    assert len(body["artifact_identity"]["clip_sha256"]) == 64
    assert body["artifact_identity"]["source_content_sha256"] == "0" * 64
    assert (
        body["dynamics_admission"]["clip_sha256"]
        == body["artifact_identity"]["clip_sha256"]
    )
    assert body["dynamics_admission"]["source_content_sha256"] == "0" * 64
    assert "tierD" in body["dynamics_admission"]["reason"]


def test_get_reference_detail_fails_closed_on_tier_k_clip_hash_drift(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    clip_path = (
        refs_root / "g1" / "fallandgetup1_subject1" / "clip.npz"
    )
    clip_path.write_bytes(clip_path.read_bytes() + b"tampered")

    response = client.get(
        "/references/fallandgetup1_subject1", params={"robot": "g1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["artifact_identity"]["verified"] is False
    assert body["artifact_identity"]["clip_sha256"] is None
    assert body["artifact_identity"]["source_content_sha256"] == "0" * 64
    assert body["dynamics_admission"]["admitted"] is False
    assert body["dynamics_admission"]["clip_sha256"] is None
    assert "does not match" in body["dynamics_admission"]["reason"]


def test_reference_artifact_routes_require_robot_identity(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    for suffix in ("", "/preview", "/file/clip.npz", "/modes"):
        response = client.get(f"/references/fallandgetup1_subject1{suffix}")
        assert response.status_code == 422, (suffix, response.text)
        assert response.json()["detail"][0]["loc"] == ["query", "robot"]


def _detail_tierd_certificate(
    refs_root: Path,
    clip_sha256: str,
    *,
    execution_contract_sha256: str = "4" * 64,
):
    from sculptor.refs.track import (
        TIER_D_CERTIFICATION_SCOPE,
        TierDCertificate,
    )
    from sculptor.policy_contract import contract_fingerprint
    from sculptor.reference_clock import build_reference_clock

    clock = build_reference_clock(
        clip_id="fallandgetup1_subject1",
        robot="g1",
        target_sha256="6" * 64,
        phase_mode="hold",
        phase_duration_s=2.0,
        n_phase_targets=8,
    )
    return TierDCertificate(
        robot="g1",
        clip_id="fallandgetup1_subject1",
        tracked_at="2026-08-17T00:00:00Z",
        iterations=500,
        mean_joint_err_rad=0.1,
        max_joint_err_rad=0.3,
        root_z_rmse_m=0.02,
        common_joint_names=("hip",),
        static_baseline_err_rad=0.2,
        static_baseline_ratio=0.5,
        rollout_path=(
            refs_root / "g1" / "fallandgetup1_subject1" / "tierD_rollout.npz"
        ),
        rollout_sha256="1" * 64,
        clip_content_sha256=clip_sha256,
        certification_scope=TIER_D_CERTIFICATION_SCOPE,
        execution_contract={"reference": {"clock_contract": clock}},
        execution_contract_sha256=execution_contract_sha256,
        execution_boundary_sha256="5" * 64,
        certificate_sha256="3" * 64,
    ), contract_fingerprint(clock)


def test_get_reference_detail_exposes_reverified_tracking_receipt_and_scope(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    from sculptor.refs.track import TIER_D_CERTIFICATION_SCOPE

    _seed_library(refs_root)
    clip_sha256 = hashlib.sha256(
        (
            refs_root
            / "g1"
            / "fallandgetup1_subject1"
            / "clip.npz"
        ).read_bytes()
    ).hexdigest()
    certificate, reference_clock_sha256 = _detail_tierd_certificate(
        refs_root, clip_sha256,
    )
    monkeypatch.setattr(
        "sculptor.refs.track.verify_tierd_certificate",
        lambda robot, clip_id: (certificate, None),
    )

    response = client.get(
        "/references/fallandgetup1_subject1", params={"robot": "g1"}
    )
    assert response.status_code == 200, response.text
    admission = response.json()["dynamics_admission"]
    assert admission["admitted"] is True
    assert admission["tier"] == "D"
    assert admission["clip_sha256"] == clip_sha256
    assert admission["source_content_sha256"] == "0" * 64
    assert admission["artifact_hash_verified"] is True
    assert admission["rollout_sha256"] == "1" * 64
    assert admission["execution_contract_sha256"] == "4" * 64
    assert admission["execution_boundary_sha256"] == "5" * 64
    assert admission["reference_clock_sha256"] == reference_clock_sha256
    assert len(admission["certificate_digest"]) == 64
    assert admission["certification_scope"] == TIER_D_CERTIFICATION_SCOPE
    assert "general_dynamics_feasibility" in (
        admission["certification_scope"]["not_certified"]
    )


def test_reference_detail_rejects_malformed_execution_identity(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    _seed_library(refs_root)
    clip_sha256 = hashlib.sha256(
        (
            refs_root
            / "g1"
            / "fallandgetup1_subject1"
            / "clip.npz"
        ).read_bytes()
    ).hexdigest()
    certificate, _clock_sha256 = _detail_tierd_certificate(
        refs_root,
        clip_sha256,
        execution_contract_sha256="malformed",
    )
    monkeypatch.setattr(
        "sculptor.refs.track.verify_tierd_certificate",
        lambda robot, clip_id: (certificate, None),
    )

    response = client.get(
        "/references/fallandgetup1_subject1", params={"robot": "g1"}
    )

    assert response.status_code == 200, response.text
    admission = response.json()["dynamics_admission"]
    assert admission["admitted"] is False
    assert admission["execution_contract_sha256"] is None
    assert admission["execution_boundary_sha256"] is None
    assert admission["reference_clock_sha256"] is None
    assert "exact execution receipt" in admission["reason"]


def test_reference_detail_rejects_tier_d_receipt_after_clip_hash_drift(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    from sculptor.refs.track import (
        TIER_D_CERTIFICATION_SCOPE,
        TierDCertificate,
    )

    _seed_library(refs_root)
    clip_path = (
        refs_root / "g1" / "fallandgetup1_subject1" / "clip.npz"
    )
    certified_sha = hashlib.sha256(clip_path.read_bytes()).hexdigest()
    certificate = TierDCertificate(
        robot="g1",
        clip_id="fallandgetup1_subject1",
        tracked_at="2026-08-17T00:00:00Z",
        iterations=500,
        mean_joint_err_rad=0.1,
        max_joint_err_rad=0.3,
        root_z_rmse_m=0.02,
        common_joint_names=("hip",),
        static_baseline_err_rad=0.2,
        static_baseline_ratio=0.5,
        rollout_path=refs_root / "g1" / "fallandgetup1_subject1" / "tierD_rollout.npz",
        rollout_sha256="1" * 64,
        clip_content_sha256=certified_sha,
        certification_scope=TIER_D_CERTIFICATION_SCOPE,
        execution_contract={"schema": "test-only"},
        certificate_sha256="3" * 64,
    )
    monkeypatch.setattr(
        "sculptor.refs.track.verify_tierd_certificate",
        lambda robot, clip_id: (certificate, None),
    )
    clip_path.write_bytes(clip_path.read_bytes() + b"tampered")

    response = client.get(
        "/references/fallandgetup1_subject1", params={"robot": "g1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["artifact_identity"]["verified"] is False
    assert body["dynamics_admission"]["admitted"] is False
    assert body["dynamics_admission"]["certificate_digest"] is None


def test_get_reference_detail_404_unknown_clip(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    r = client.get("/references/no-such-clip", params={"robot": "g1"})
    assert r.status_code == 404


def test_get_reference_detail_404_invalid_clip_id_regex(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    # Uppercase + traversal-shaped path segments both fail the
    # `^[a-z0-9][a-z0-9_-]{0,95}$` guard.
    for bad_id in ["Bad-ID", "..", "has space"]:
        r = client.get(f"/references/{bad_id}", params={"robot": "g1"})
        assert r.status_code == 404, (bad_id, r.text)


# ── GET /references/{clip_id}/preview ────────────────────────────────
def test_preview_404_when_absent(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root)  # no previews written
    r = client.get(
        "/references/fallandgetup1_subject1/preview", params={"robot": "g1"}
    )
    assert r.status_code == 404


def test_preview_200_when_present(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root, with_preview_for={"fallandgetup1_subject1"})
    r = client.get(
        "/references/fallandgetup1_subject1/preview", params={"robot": "g1"}
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"


# ── GET /references/{clip_id}/file/clip.npz ──────────────────────────
def test_download_clip_file(client: TestClient, refs_root: Path) -> None:
    _seed_library(refs_root)
    r = client.get(
        "/references/fallandgetup1_subject1/file/clip.npz",
        params={"robot": "g1"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    assert len(r.content) > 0


def test_download_clip_file_404_unknown_clip(
    client: TestClient, refs_root: Path,
) -> None:
    _seed_library(refs_root)
    r = client.get(
        "/references/no-such-clip/file/clip.npz", params={"robot": "g1"}
    )
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


def test_get_reference_detail_preview_and_download_use_exact_t1_pair(
    client: TestClient, refs_root: Path,
) -> None:
    """Every artifact route resolves the explicit robot/clip identity."""
    _seed_library(refs_root)
    _seed_t1_clip(refs_root, with_preview=True)
    clip_id = "fallandgetup1_subject1_t1"

    r = client.get(f"/references/{clip_id}", params={"robot": "t1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["index_row"]["robot"] == "t1"
    assert body["provenance"]["robot"] == "t1"

    r2 = client.get(f"/references/{clip_id}/preview", params={"robot": "t1"})
    assert r2.status_code == 200, r2.text
    assert r2.headers["content-type"] == "image/png"

    r3 = client.get(
        f"/references/{clip_id}/file/clip.npz", params={"robot": "t1"}
    )
    assert r3.status_code == 200, r3.text
    assert len(r3.content) > 0


def test_duplicate_clip_ids_resolve_by_exact_robot_pair(
    client: TestClient, refs_root: Path,
) -> None:
    """A clip ID is never a globally unique artifact identity."""
    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root, with_preview_for={clip_id})
    _seed_t1_clip(refs_root, with_preview=True, clip_id=clip_id)

    for robot in ("g1", "t1"):
        detail = client.get(
            f"/references/{clip_id}", params={"robot": robot}
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["index_row"]["robot"] == robot
        assert detail.json()["provenance"]["robot"] == robot
        assert client.get(
            f"/references/{clip_id}/preview", params={"robot": robot}
        ).status_code == 200
        assert client.get(
            f"/references/{clip_id}/file/clip.npz", params={"robot": robot}
        ).status_code == 200

    assert client.get(
        f"/references/{clip_id}", params={"robot": "go1"}
    ).status_code == 404


# ── attach / detach ─────────────────────────────────────────────────
def test_attach_reference_sets_stage_fields(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root, with_preview_for={"fallandgetup1_subject1"})
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _certify_clip(
        refs_root, "g1", "fallandgetup1_subject1", project_dir=project_dir,
    )
    _write_mission(project_dir, "m1", [_stage_dict("torso_righting")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/torso_righting/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reference_clip_id"] == "fallandgetup1_subject1"
    assert body["reference_tier"] == "D"
    assert body["reference_match_confidence"] is None
    assert body["reference_robot"] == "g1"
    assert len(body["reference_clip_sha256"]) == 64
    assert len(body["reference_certificate_sha256"]) == 64
    assert len(body["reference_execution_contract_sha256"]) == 64
    assert len(body["reference_execution_boundary_sha256"]) == 64

    # Persisted to mission.json.
    mission_json = json.loads((project_dir / ".missions" / "m1" / "mission.json").read_text())
    stage = mission_json["stages"][0]
    assert stage["reference_clip_id"] == "fallandgetup1_subject1"
    assert stage["reference_tier"] == "D"
    assert stage["reference_match_confidence"] is None
    assert stage["reference_robot"] == "g1"
    assert stage["reference_clip_sha256"] == body["reference_clip_sha256"]
    assert (
        stage["reference_certificate_sha256"]
        == body["reference_certificate_sha256"]
    )
    assert (
        stage["reference_execution_contract_sha256"]
        == body["reference_execution_contract_sha256"]
    )
    assert (
        stage["reference_execution_boundary_sha256"]
        == body["reference_execution_boundary_sha256"]
    )

    # And it flows through the mission GET (StageSchema mirror).
    r2 = client.get(f"/projects/{slug}/missions/m1")
    assert r2.status_code == 200, r2.text
    stage2 = r2.json()["stages"][0]
    assert stage2["reference_clip_id"] == "fallandgetup1_subject1"
    assert stage2["reference_tier"] == "D"
    assert stage2["reference_robot"] == "g1"


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
    assert r.status_code == 412, r.text

    mission_json = json.loads((project_dir / ".missions" / "m1" / "mission.json").read_text())
    stage = mission_json["stages"][0]
    assert stage.get("reference_clip_id") is None


def test_attach_reference_rejects_missing_tierd_certificate(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a")])

    response = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": "fallandgetup1_subject1"},
    )

    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/reference-feasibility"
    stage = json.loads(
        (project_dir / ".missions" / "m1" / "mission.json").read_text()
    )["stages"][0]
    assert stage.get("reference_clip_id") is None


@pytest.mark.parametrize("tamper", ["clip_bytes", "rollout_hash"])
def test_attach_reference_rejects_stale_or_forged_certificate(
    client: TestClient,
    refs_root: Path,
    tmp_projects_root: Path,
    tamper: str,
) -> None:
    from sculptor.refs import library

    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root)
    slug = _make_project(client, name=f"tamper {tamper}")
    project_dir = tmp_projects_root / slug
    _certify_clip(refs_root, "g1", clip_id, project_dir=project_dir)
    clip_dir = library.clip_dir("g1", clip_id, root=refs_root)
    if tamper == "clip_bytes":
        (clip_dir / library.CLIP_FILENAME).write_bytes(b"stale clip bytes")
    else:
        provenance = library.read_provenance("g1", clip_id, root=refs_root)
        provenance["tierD"]["rollout_sha256"] = "f" * 64
        library.write_provenance(
            "g1", clip_id, provenance, root=refs_root,
        )

    _write_mission(project_dir, "m1", [_stage_dict("a")])
    response = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": clip_id},
    )

    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/reference-feasibility"


def test_attach_reference_uses_project_robot_with_duplicate_clip_ids(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    from sculptor.refs import library

    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root)
    t1_dir = library.clip_dir("t1", clip_id, root=refs_root)
    t1_dir.mkdir(parents=True, exist_ok=True)
    _write_clip_npz(t1_dir)
    _write_provenance(
        t1_dir,
        clip_id=clip_id,
        robot="t1",
        text="same id, different robot",
        labels=["duplicate"],
    )
    slug = _make_project(client, name="compound identity")
    project_dir = tmp_projects_root / slug
    _certify_clip(refs_root, "g1", clip_id, project_dir=project_dir)
    library.rebuild_index(root=refs_root)
    _write_mission(project_dir, "m1", [_stage_dict("a")])
    response = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": clip_id},
    )

    assert response.status_code == 200, response.text
    assert response.json()["reference_robot"] == "g1"
    stage = json.loads(
        (project_dir / ".missions" / "m1" / "mission.json").read_text()
    )["stages"][0]
    assert stage["reference_robot"] == "g1"


def test_mission_run_api_rejects_reference_changed_after_attach(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    from sculptor.refs import library

    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root)
    slug = _make_project(client, name="stale before enqueue")
    project_dir = tmp_projects_root / slug
    _certify_clip(refs_root, "g1", clip_id, project_dir=project_dir)
    _write_mission(project_dir, "m1", [_stage_dict("a")])
    attached = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": clip_id},
    )
    assert attached.status_code == 200, attached.text
    clip_path = (
        library.clip_dir("g1", clip_id, root=refs_root)
        / library.CLIP_FILENAME
    )
    clip_path.write_bytes(b"changed after stage attachment")

    response = client.post(f"/projects/{slug}/missions/m1/run", json={})

    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/reference-feasibility"


def test_mission_run_api_rejects_target_contract_drift_after_attach(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """A donor certificate cannot authorize a newly different target task."""
    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root)
    slug = _make_project(client, name="target drift before enqueue")
    project_dir = tmp_projects_root / slug
    _certify_clip(refs_root, "g1", clip_id, project_dir=project_dir)
    _write_mission(project_dir, "m1", [_stage_dict("a")])
    attached = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": clip_id},
    )
    assert attached.status_code == 200, attached.text

    client.app.state.project_store.set_adapter_section(  # type: ignore[attr-defined]
        slug,
        "sculptor.adapters.mjlab.MjlabAdapter",
        {"task_id": "Mjlab-Velocity-Rough-Unitree-G1"},
    )
    response = client.post(f"/projects/{slug}/missions/m1/run", json={})

    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/reference-feasibility"
    assert "identity.task_id differs" in response.json()["detail"]


def test_mission_run_worker_rechecks_before_subprocess_spawn(
    client: TestClient,
    refs_root: Path,
    tmp_projects_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.job_manager import Job
    from backend.services.mission_jobs import run_mission_execute_job
    from sculptor.refs import library

    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root)
    slug = _make_project(client, name="stale before spawn")
    project_dir = tmp_projects_root / slug
    _certify_clip(refs_root, "g1", clip_id, project_dir=project_dir)
    _write_mission(project_dir, "m1", [_stage_dict("a")])
    attached = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": clip_id},
    )
    assert attached.status_code == 200, attached.text
    provenance = library.read_provenance("g1", clip_id, root=refs_root)
    provenance["tierD"]["tracked_at"] = "forged-after-queue"
    library.write_provenance("g1", clip_id, provenance, root=refs_root)

    spawned = False

    async def _unexpected_spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("training subprocess must not be spawned")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _unexpected_spawn)
    runner = run_mission_execute_job(
        project_dir=project_dir,
        project_slug=slug,
        mission_slug="m1",
    )
    job = Job(
        job_id="job-tierd-boundary",
        kind="mission_execute",
        project_slug=slug,
        status="running",
    )

    with pytest.raises(RuntimeError, match="admission failed before spawn"):
        asyncio.run(runner(job, asyncio.Event()))
    assert spawned is False
    assert any(
        event["type"] == "mission_reference_admission_failed"
        for event in job.events
    )


def test_mission_run_worker_rejects_target_contract_drift_before_spawn(
    client: TestClient,
    refs_root: Path,
    tmp_projects_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.job_manager import Job
    from backend.services.mission_jobs import run_mission_execute_job

    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root)
    slug = _make_project(client, name="target drift before spawn")
    project_dir = tmp_projects_root / slug
    _certify_clip(refs_root, "g1", clip_id, project_dir=project_dir)
    _write_mission(project_dir, "m1", [_stage_dict("a")])
    attached = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": clip_id},
    )
    assert attached.status_code == 200, attached.text
    client.app.state.project_store.set_adapter_section(  # type: ignore[attr-defined]
        slug,
        "sculptor.adapters.mjlab.MjlabAdapter",
        {"task_id": "Mjlab-Velocity-Rough-Unitree-G1"},
    )

    spawned = False

    async def _unexpected_spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("training subprocess must not be spawned")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _unexpected_spawn)
    runner = run_mission_execute_job(
        project_dir=project_dir,
        project_slug=slug,
        mission_slug="m1",
    )
    job = Job(
        job_id="job-target-boundary",
        kind="mission_execute",
        project_slug=slug,
        status="running",
    )

    with pytest.raises(RuntimeError, match="admission failed before spawn"):
        asyncio.run(runner(job, asyncio.Event()))
    assert spawned is False
    failure = next(
        event for event in job.events
        if event["type"] == "mission_reference_admission_failed"
    )
    assert "identity.task_id differs" in failure["error"]


def test_mission_run_worker_rejects_exact_target_receipt_drift(
    client: TestClient,
    refs_root: Path,
    tmp_projects_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even compatible target changes cannot inherit the queued receipt."""
    from backend.services.job_manager import Job
    from backend.services.mission_jobs import run_mission_execute_job

    clip_id = "fallandgetup1_subject1"
    _seed_library(refs_root)
    slug = _make_project(client, name="exact receipt drift")
    project_dir = tmp_projects_root / slug
    _certify_clip(refs_root, "g1", clip_id, project_dir=project_dir)
    _write_mission(project_dir, "m1", [_stage_dict("a")])
    attached = client.post(
        f"/projects/{slug}/missions/m1/stages/a/reference",
        json={"clip_id": clip_id},
    )
    assert attached.status_code == 200, attached.text

    spawned = False

    async def _unexpected_spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("training subprocess must not be spawned")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _unexpected_spawn)
    runner = run_mission_execute_job(
        project_dir=project_dir,
        project_slug=slug,
        mission_slug="m1",
        reference_target_receipt={
            "schema": 1,
            "target_robot": "g1",
            "policy_contract_sha256": "f" * 64,
        },
    )
    job = Job(
        job_id="job-exact-target-boundary",
        kind="mission_execute",
        project_slug=slug,
        status="running",
    )

    with pytest.raises(RuntimeError, match="admission failed before spawn"):
        asyncio.run(runner(job, asyncio.Event()))
    assert spawned is False
    failure = next(
        event for event in job.events
        if event["type"] == "mission_reference_admission_failed"
    )
    assert "target execution contract changed after queue" in failure["error"]


def test_attach_reference_pending_stage_without_training_dir(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """§commit 8b0bfa3 precedent: a pending stage that has never
    trained has no stages/<stage>/ dir. Attach must validate against
    mission.json's stage list, not the training dir."""
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _certify_clip(
        refs_root, "g1", "fallandgetup1_subject1", project_dir=project_dir,
    )
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
    _certify_clip(
        refs_root, "g1", "fallandgetup1_subject1", project_dir=project_dir,
    )
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
    assert stage["reference_robot"] is None
    assert stage["reference_clip_sha256"] is None
    assert stage["reference_certificate_sha256"] is None
    assert stage["reference_execution_contract_sha256"] is None
    assert stage["reference_execution_boundary_sha256"] is None


def test_detach_reference_409_when_mission_job_active(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    _seed_library(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _certify_clip(
        refs_root, "g1", "fallandgetup1_subject1", project_dir=project_dir,
    )
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
    detail = client.get(
        "/references/novel-motion--g1", params={"robot": "g1"}
    )
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


# ── the OGMP-inspired phase scaffold + its reward module ───────────────
# `ModeTimeline.tsx` already draws the automaton at compose time by mirroring
# the derivation in TypeScript. These cover the half that cannot be mirrored:
# turning it into reward code (HANDOFF.md §12).
def _write_composite(root: Path, clip_id: str = "novel-jump-kick--g1",
                     robot: str = "g1", *, n: int = 240, fps: float = 120.0,
                     j: int = 6, seams=(80, 160),
                     labels=("approach", "launch", "strike")) -> Path:
    from sculptor.refs import library

    t = np.arange(n, dtype=np.float64) / fps
    d = library.clip_dir(robot, clip_id, root=root)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"clip_id": clip_id,
            "composition": {
                "seam_frames": list(seams),
                "segments": [{"index": i, "label": label,
                              "source_id": f"src_{i}", "source_fps": 60.0,
                              "source_frames": [0, 60]}
                             for i, label in enumerate(labels)]}}
    np.savez(
        d / "clip.npz",
        fps=np.float64(fps),
        root_pos_z=0.70 + 0.02 * np.sin(2 * np.pi * 0.5 * t),
        root_quat_wxyz=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        joint_pos=(0.10 * np.sin(2 * np.pi * 0.5 * t)[:, None]
                   + 0.01 * np.arange(j)[None, :]),
        joint_names=np.array([f"joint_{i}" for i in range(j)]),
        meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    return d


def test_reference_modes_are_derived_from_the_clips_own_provenance(
    client: TestClient, refs_root: Path,
) -> None:
    """One composed segment is one mode, each seam a transition — a read of
    what `refs.compose` already recorded, not a new derivation."""
    _write_composite(refs_root)
    r = client.get(
        "/references/novel-jump-kick--g1/modes", params={"robot": "g1"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fps"] == 120.0
    # Capability disclosure is part of the API contract. A phase-window
    # scaffold must never be presented as the paper's closed-loop oracle or
    # latent-mode-conditioned policy merely because both use the word mode.
    from sculptor.kg.capabilities import mode_api_capability_summary

    assert body["capability"] == mode_api_capability_summary()
    assert [m["name"] for m in body["modes"]] == ["approach", "launch", "strike"]
    assert body["modes"][0]["reward_terms"] == []
    assert body["modes"][0]["success_predicate"] is None
    assert body["modes"][0]["start_s"] == 0.0
    # `mode_phase_windows` rounds to 4 dp — 0.1 ms, five orders of magnitude
    # finer than the 20 ms control step these windows gate against.
    assert body["modes"][0]["end_s"] == pytest.approx(80 / 120.0, abs=1e-4)
    assert body["modes"][-1]["end_s"] == pytest.approx(2.0, abs=1e-4)
    assert [(t["from_mode"], t["to_mode"]) for t in body["transitions"]] == [
        ("approach", "launch"), ("launch", "strike")]


def test_a_non_composite_reference_is_422_not_500(
    client: TestClient, refs_root: Path,
) -> None:
    """The common mistake. The request is well-formed; the clip just is not a
    composite, so there is one mode and no transition to derive."""
    _seed_library(refs_root)
    r = client.get(
        "/references/walk1_subject2/modes", params={"robot": "g1"}
    )
    assert r.status_code == 422, r.text
    assert "composition" in r.json()["detail"]


def test_modes_for_a_malformed_clip_id_is_404(client: TestClient) -> None:
    assert client.get(
        "/references/..%2Fetc/modes", params={"robot": "g1"}
    ).status_code in (404, 400)


def test_scaffolding_a_mode_reward_writes_it_into_the_project(
    client: TestClient, refs_root: Path, tmp_path: Path,
) -> None:
    _write_composite(refs_root)
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/references/novel-jump-kick--g1/mode-reward",
        json={"clip_id": "novel-jump-kick--g1", "robot": "g1",
              "goal": "run in and strike at the apex"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["unauthored"] == ["approach", "launch", "strike"]
    assert all(m["authored"] is False for m in body["modes"])

    written = Path(body["path"])
    assert written.is_file() and written.name == "mode_reward_v0.py"
    src = written.read_text(encoding="utf-8")
    assert "def compute_reward_batched(" in src
    assert "run in and strike at the apex" in src
    # Tracking is on by default — without it the module pays zero until every
    # mode is authored, and nothing tells the policy to follow the reference.
    assert "TARGET_JOINT_POS" in src
    assert body["tracking"] is True
    listing = client.get(f"/projects/{slug}/mode-rewards").json()
    assert listing["mode_rewards"][0]["tracking_enabled"] is True


def test_scaffolding_without_tracking_omits_the_backbone(
    client: TestClient, refs_root: Path,
) -> None:
    _write_composite(refs_root)
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/references/novel-jump-kick--g1/mode-reward",
        json={
            "clip_id": "novel-jump-kick--g1",
            "robot": "g1",
            "tracking": False,
        })
    assert r.status_code == 200, r.text
    assert "TARGET_JOINT_POS" not in Path(r.json()["path"]).read_text()
    listing = client.get(f"/projects/{slug}/mode-rewards").json()
    assert listing["mode_rewards"][0]["tracking_enabled"] is False


def test_scaffolding_twice_is_a_409_unless_overwrite(
    client: TestClient, refs_root: Path,
) -> None:
    """Regenerating discards authored mode bodies. The scaffold is the cheap
    half; the authored terms are the expensive one."""
    _write_composite(refs_root)
    slug = _make_project(client)
    url = f"/projects/{slug}/references/novel-jump-kick--g1/mode-reward"
    payload = {"clip_id": "novel-jump-kick--g1", "robot": "g1"}
    assert client.post(url, json=payload).status_code == 200
    again = client.post(url, json=payload)
    assert again.status_code == 409, again.text
    assert "overwrite" in again.json()["detail"]
    assert client.post(url, json={**payload, "overwrite": True}).status_code == 200


def test_a_filename_cannot_escape_the_rewards_directory(
    client: TestClient, refs_root: Path,
) -> None:
    """`filename` arrives in a request body, so it is validated rather than
    sanitized into something that merely looks accepted."""
    _write_composite(refs_root)
    slug = _make_project(client)
    url = f"/projects/{slug}/references/novel-jump-kick--g1/mode-reward"
    for bad in ("../escape.py", "sub/dir.py", "no_extension", ".hidden.py",
                "/abs/path.py"):
        r = client.post(url, json={"clip_id": "novel-jump-kick--g1",
                                   "robot": "g1",
                                   "filename": bad})
        assert r.status_code == 422, f"{bad!r} was accepted: {r.text}"


def test_a_clip_id_mismatch_between_path_and_body_is_refused(
    client: TestClient, refs_root: Path,
) -> None:
    _write_composite(refs_root)
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/references/novel-jump-kick--g1/mode-reward",
        json={"clip_id": "something-else--g1", "robot": "g1"})
    assert r.status_code == 422, r.text


def test_scaffolding_for_an_unknown_project_is_404(
    client: TestClient, refs_root: Path,
) -> None:
    _write_composite(refs_root)
    r = client.post(
        "/projects/no-such-project/references/novel-jump-kick--g1/mode-reward",
        json={"clip_id": "novel-jump-kick--g1", "robot": "g1"})
    assert r.status_code == 404, r.text


# ── authoring one mode ────────────────────────────────────────────────────
AUTHOR_URL = "/projects/{slug}/references/novel-jump-kick--g1/mode-reward/author"


def _scaffold(client: TestClient, slug: str, overwrite: bool = False) -> dict:
    r = client.post(
        f"/projects/{slug}/references/novel-jump-kick--g1/mode-reward",
        json={
            "clip_id": "novel-jump-kick--g1",
            "robot": "g1",
            "overwrite": overwrite,
        })
    assert r.status_code == 200, r.text
    return r.json()


def _author_body(**kw) -> dict:
    return {
        "clip_id": "novel-jump-kick--g1",
        "robot": "g1",
        "mode": "launch",
        **kw,
    }


def test_authoring_a_mode_fires_a_job_and_chains_the_filename(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    """The whole route minus Claude. What is asserted is what the route
    decides: which file is read, which is written, and that it is one job."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    scaffold = _scaffold(client, slug)

    captured: dict = {}

    def _fake_job(**kwargs):
        captured.update(kwargs)

        async def _runner(job, cancel):
            return {"mode": kwargs["mode"], "authored_count": 1,
                    "mode_count": 3, "pending": ["approach", "strike"]}
        return _runner

    monkeypatch.setattr("backend.services.mode_jobs.run_mode_author_job",
                        _fake_job)
    r = client.post(AUTHOR_URL.format(slug=slug), json=_author_body())
    assert r.status_code == 202, r.text
    assert r.json()["kind"] == "mode_author"

    assert captured["mode"] == "launch"
    assert Path(captured["reward_path"]).name == "mode_reward_v0.py"
    # Chained, not overwritten: the scaffold survives a bad edit.
    assert Path(captured["out_path"]).name == "mode_reward_v1.py"
    assert Path(scaffold["path"]).is_file()


def test_authoring_an_unknown_mode_is_422_before_any_model_call(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    r = client.post(AUTHOR_URL.format(slug=slug),
                    json=_author_body(mode="nosuchmode"))
    assert r.status_code == 422, r.text
    assert "approach, launch, strike" in r.json()["detail"]


def test_authoring_without_a_scaffold_is_404(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    """There is nothing to author INTO — the per-mode gating comes from the
    scaffold, not from the model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    r = client.post(AUTHOR_URL.format(slug=slug), json=_author_body())
    assert r.status_code == 404, r.text
    assert "scaffold" in r.json()["detail"]


def test_authoring_in_place_is_refused(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    r = client.post(AUTHOR_URL.format(slug=slug),
                    json=_author_body(out_filename="mode_reward_v0.py"))
    assert r.status_code == 422, r.text
    assert "out_filename" in r.json()["title"]


def test_authoring_rejects_a_filename_that_escapes_the_project(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    for name in ("../../etc/passwd.py", "sub/dir.py", ".hidden.py", "no_ext"):
        r = client.post(AUTHOR_URL.format(slug=slug),
                        json=_author_body(out_filename=name))
        assert r.status_code == 422, f"{name} -> {r.status_code}"


def test_authoring_without_an_api_key_is_503_not_a_wedged_job(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    r = client.post(AUTHOR_URL.format(slug=slug), json=_author_body())
    assert r.status_code == 503, r.text


def test_a_second_authoring_job_is_refused_while_one_is_running(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    """Modes are authored one at a time; two jobs would race on the chained
    file and the second would author into a scaffold that is about to move."""
    import asyncio

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)

    def _slow_job(**_kw):
        async def _runner(job, cancel):
            await asyncio.sleep(30)
            return {}
        return _runner

    monkeypatch.setattr("backend.services.mode_jobs.run_mode_author_job",
                        _slow_job)
    first = client.post(AUTHOR_URL.format(slug=slug), json=_author_body())
    assert first.status_code == 202, first.text
    second = client.post(AUTHOR_URL.format(slug=slug),
                         json=_author_body(mode="approach"))
    assert second.status_code == 409, second.text
    assert second.json()["active_job_id"] == first.json()["job_id"]


def test_authoring_for_a_non_composite_reference_is_422(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _seed_library(refs_root)
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/references/walk1_subject2/mode-reward/author",
        json={"clip_id": "walk1_subject2", "robot": "g1", "mode": "whole"})
    assert r.status_code == 422, r.text


# ── promotion: the step that makes a run actually use the reward ─────────
PROMOTE_URL = ("/projects/{slug}/references/novel-jump-kick--g1"
               "/mode-reward/promote")


def _patch_adapter(monkeypatch):
    """Stand in for the project's adapter so the mjlab-shaped reward contract
    is available without an mjlab environment. The authoring job loads it for
    real, which is the point — this only removes the env dependency."""
    import types

    contract = types.SimpleNamespace(
        supports_batched=True,
        state_schema={"qpos": (29,), "projected_gravity_b": (3,),
                      "actuator_force": (29,)},
        info_schema={"episode_length": (), "step_dt": (), "base_height": ()},
        expected_info_keys=["episode_length", "step_dt", "base_height"])
    monkeypatch.setattr(
        "sculptor.adapters.base.load_adapter",
        lambda _p: types.SimpleNamespace(reward_contract=lambda: contract))
    return contract


def _author_all(client: TestClient, slug: str, monkeypatch) -> str:
    """Author every mode, returning the final filename."""
    from sculptor.mode_rewards import MODE_FN_PREFIX

    _patch_adapter(monkeypatch)

    name = "mode_reward_v0.py"
    for i, mode in enumerate(("approach", "launch", "strike"), start=1):
        def _edit(*, current_reward_path, new_iter_id, _m=mode, **_kw):
            src = Path(current_reward_path).read_text(encoding="utf-8")
            for suffix, sig, ret in (
                ("", "(state, action, next_state, info)", "0.5"),
                ("_batched", "(state, action, next_state, info, like)",
                 "like + 0.5"),
            ):
                fn = f"{MODE_FN_PREFIX}{_m}{suffix}"
                head = f"def {fn}{sig}:"
                j = src.index(head)
                k = src.index("\ndef ", j)
                src = (src[:j] + head + "\n    del state, action, next_state, info\n"
                       + f"    v = {ret}\n    return v, {{'{_m}_core': v}}\n"
                       + src[k:])
            dest = Path(current_reward_path).parent / f"{new_iter_id}.py"
            dest.write_text(src, encoding="utf-8")
            return dest
        monkeypatch.setattr("sculptor.edit.apply_prompt_edit", _edit)
        r = client.post(AUTHOR_URL.format(slug=slug),
                        json={"clip_id": "novel-jump-kick--g1",
                              "robot": "g1", "mode": mode,
                              "filename": name})
        assert r.status_code == 202, r.text
        # Authoring is a background job; the next call reads the file it
        # writes, so it has to actually be finished.
        _await_job(client, r.json()["job_id"])
        name = f"mode_reward_v{i}.py"
    return name


def _await_job(client: TestClient, job_id: str, tries: int = 200) -> dict:
    import time

    for _ in range(tries):
        d = client.get(f"/jobs/{job_id}").json()
        if d["status"] in ("completed", "errored", "stopped"):
            assert d["status"] == "completed", d.get("error")
            return d
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_promoting_makes_the_authored_reward_the_one_a_run_trains(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    """The step without which the whole feature silently does nothing:
    `mode_reward_v3.py` is not a version, and `current.py` — what every
    adapter imports — points at a `v<n>.py`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    final = _author_all(client, slug, monkeypatch)

    before = client.get(f"/projects/{slug}/rewards").json()
    assert [v["version"] for v in before] == [0], "authoring alone adds no version"

    r = client.post(PROMOTE_URL.format(slug=slug), json={"filename": final})
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 1
    assert r.json()["unauthored"] == []

    after = client.get(f"/projects/{slug}/rewards").json()
    assert [v["version"] for v in after] == [1, 0]
    promoted = client.get(f"/projects/{slug}/rewards/1").json()
    assert promoted["spec"]["version"] == "v1"
    assert "def compute_reward_batched(" in promoted["source"]


def test_promoting_a_half_authored_reward_is_refused(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    r = client.post(PROMOTE_URL.format(slug=slug),
                    json={"filename": "mode_reward_v0.py"})
    assert r.status_code == 409, r.text
    d = r.json()["detail"]
    assert "unauthored stub" in d and "approach" in d
    assert [v["version"] for v in client.get(f"/projects/{slug}/rewards").json()] == [0]


def test_a_bare_scaffold_can_be_promoted_deliberately(
    client: TestClient, refs_root: Path, monkeypatch,
) -> None:
    """The tracking backbone alone is trainable — that IS the Tier-D path — so
    the refusal is a flag, not a wall."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    r = client.post(PROMOTE_URL.format(slug=slug),
                    json={"filename": "mode_reward_v0.py",
                          "allow_unauthored": True})
    assert r.status_code == 200, r.text
    assert sorted(r.json()["unauthored"]) == ["approach", "launch", "strike"]
    assert [v["version"] for v in client.get(f"/projects/{slug}/rewards").json()] == [1, 0]


def test_same_clip_id_with_changed_bytes_is_stale_not_promotable(
    client: TestClient, refs_root: Path,
) -> None:
    """The id is a label, not provenance. Replacing bytes under it must not
    reuse phase windows/rewards that were reviewed against the old motion."""
    clip_dir = _write_composite(refs_root)
    slug = _make_project(client)
    body = _scaffold(client, slug)

    before = client.get(f"/projects/{slug}/mode-rewards").json()
    mine = next(
        item for item in before["mode_rewards"]
        if item["filename"] == body["filename"]
    )
    assert mine["context_current"] is True
    assert mine["execution_context_digest"] == body["execution_context_digest"]

    # A ZIP reader permits trailing bytes, so the fixture remains readable;
    # its content identity nonetheless changed.
    with (clip_dir / "clip.npz").open("ab") as stream:
        stream.write(b"changed-reference-bytes")

    after = client.get(f"/projects/{slug}/mode-rewards").json()
    mine = next(
        item for item in after["mode_rewards"]
        if item["filename"] == body["filename"]
    )
    assert mine["context_current"] is False

    promoted = client.post(
        PROMOTE_URL.format(slug=slug),
        json={"filename": body["filename"], "allow_unauthored": True},
    )
    assert promoted.status_code == 409, promoted.text
    assert "older execution context" in promoted.json()["title"]


def test_legacy_mode_reward_without_source_robot_fails_closed(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """Missing provenance must not be silently interpreted as G1.

    Older phase rewards did not always bind a source robot.  The target
    project's robot is not evidence about the artifact that produced those
    phase windows, so such a reward remains inspectable but cannot be
    promoted until it is regenerated.
    """
    from sculptor.mode_rewards import _rewrite_reward_spec

    _write_composite(refs_root)
    slug = _make_project(client)
    body = _scaffold(client, slug)
    reward_path = tmp_projects_root / slug / "rewards" / body["filename"]
    reward_path.write_text(
        _rewrite_reward_spec(
            reward_path.read_text(encoding="utf-8"),
            {"reference_robot": ""},
        ),
        encoding="utf-8",
    )

    listing = client.get(f"/projects/{slug}/mode-rewards").json()
    mine = next(
        item for item in listing["mode_rewards"]
        if item["filename"] == body["filename"]
    )
    assert mine["reference_robot"] == ""
    assert mine["context_current"] is False
    assert "source robot identity is missing" in mine["context_blocker"]

    refused = client.post(
        PROMOTE_URL.format(slug=slug),
        json={"filename": body["filename"], "allow_unauthored": True},
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["title"] == "mode reward source robot is unknown"


def test_promoting_a_filename_that_escapes_the_project_is_refused(
    client: TestClient, refs_root: Path,
) -> None:
    _write_composite(refs_root)
    slug = _make_project(client)
    for name in ("../../etc/passwd.py", "sub/dir.py", "nope"):
        r = client.post(PROMOTE_URL.format(slug=slug), json={"filename": name})
        assert r.status_code == 422, f"{name} -> {r.status_code}"


def test_promoting_a_missing_file_is_404(
    client: TestClient, refs_root: Path,
) -> None:
    _write_composite(refs_root)
    slug = _make_project(client)
    r = client.post(PROMOTE_URL.format(slug=slug),
                    json={"filename": "mode_reward_v9.py"})
    assert r.status_code == 404, r.text


def test_mode_rewards_reports_nothing_promoted_until_promotion(
    client: TestClient, refs_root: Path, tmp_projects_root: Path, monkeypatch,
) -> None:
    """Authored on disk is not the same as trained.

    Both states rendered identically in the UI: the panel listed
    `mode_reward_v<n>.py` either way, and the Rewards tab — which matches
    `^v(\\d+)\\.py$` — could not see the file at all. So a user who authored
    four modes and never pressed Promote saw four green modes over a run that
    trained the starter reward.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    _author_all(client, slug, monkeypatch)

    body = client.get(f"/projects/{slug}/mode-rewards").json()

    assert [f["filename"] for f in body["mode_rewards"]], "authoring wrote files"
    assert body["promoted"] is None


def test_mode_rewards_promoted_names_what_a_run_would_train(
    client: TestClient, refs_root: Path, tmp_projects_root: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    final = _author_all(client, slug, monkeypatch)
    assert client.post(PROMOTE_URL.format(slug=slug),
                       json={"filename": final}).status_code == 200

    promoted = client.get(f"/projects/{slug}/mode-rewards").json()["promoted"]

    assert promoted is not None
    assert promoted["version"] == 1
    assert promoted["filename"] == "v1.py"
    assert promoted["clip_id"] == "novel-jump-kick--g1"
    assert promoted["unauthored"] == []
    assert len(promoted["modes"]) >= 1
    assert all(m["authored"] for m in promoted["modes"])


def test_mode_rewards_resolves_current_pointer_not_highest_version(
    client: TestClient, refs_root: Path, tmp_projects_root: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    final = _author_all(client, slug, monkeypatch)
    assert client.post(
        PROMOTE_URL.format(slug=slug), json={"filename": final}
    ).status_code == 200
    rewards = tmp_projects_root / slug / "rewards"
    # Keep-best can leave later files on disk while current.py selects v1.
    (rewards / "v4.py").write_bytes((rewards / "v1.py").read_bytes())

    promoted = client.get(f"/projects/{slug}/mode-rewards").json()["promoted"]

    assert promoted is not None
    assert promoted["version"] == 1
    assert promoted["filename"] == "v1.py"


def test_mode_rewards_promoted_clears_when_a_flat_reward_supersedes_it(
    client: TestClient, refs_root: Path, tmp_projects_root: Path, monkeypatch,
) -> None:
    """A later flat version means the modes stopped being what trains.

    This is the state a reference-guided run used to leave behind silently:
    it writes the next `v<n>.py` and repoints `current.py`, and the authored
    mode bodies stay on disk looking untouched.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    final = _author_all(client, slug, monkeypatch)
    client.post(PROMOTE_URL.format(slug=slug), json={"filename": final})

    v2 = tmp_projects_root / slug / "rewards" / "v2.py"
    v2.write_text(
        "REWARD_SPEC = {'version': 'v2'}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 0.0, {}\n",
        encoding="utf-8")
    from sculptor.edit import _write_current_reexport
    _write_current_reexport(v2.parent, v2)

    body = client.get(f"/projects/{slug}/mode-rewards").json()

    assert body["promoted"] is None
    assert [f["filename"] for f in body["mode_rewards"]], "the files remain"


# ── the mission a per-mode reward is authored against ───────────────────
def _write_selection(project_dir: Path, *, episode_s: float = 20.0,
                     with_goal: bool = True) -> None:
    """The three files `_selection_specs` reads, as a promoted selection."""
    env = project_dir / "env"
    env.mkdir(parents=True, exist_ok=True)
    goal = {"id": "complete_course", "type": "waypoint_sequence",
            "success": {"predicate": "sequence_complete", "hold_s": 0.15}}
    (env / "task.json").write_text(json.dumps({"shared": {
        "goal": goal if with_goal else {},
        "termination": {"episode_length_s": episode_s}}}), encoding="utf-8")
    (env / "world.json").write_text(json.dumps({"shared": {"obstacles": {
        "layout": "linear", "start_offset_m": 0.81,
        "course": [{"element": "platform", "id": "box_01",
                    "nominal": {"height_m": 0.231}}]}}}), encoding="utf-8")
    (env / "catalog.json").write_text(json.dumps({"channels": [
        {"name": "goal__complete_course__waypoint_distance",
         "access": "shared_shaping", "metric_role": "progress",
         "source": {"goal": "complete_course"}},
        {"name": "goal__complete_course__waypoint_index",
         "access": "metric_only", "metric_role": "progress",
         "source": {"goal": "complete_course"}}]}), encoding="utf-8")
    (env / "selection_current.json").write_text(json.dumps({"refs": {
        "task": {"path": "env/task.json"},
        "world": {"path": "env/world.json"},
        "channel_catalog": {"path": "env/catalog.json"}}}), encoding="utf-8")


def test_a_project_with_no_selection_stays_on_clip_time_and_full_tracking(
    tmp_projects_root: Path,
) -> None:
    """Pure imitation must generate exactly what it always did."""
    from backend.routes.references import _episode_horizon_s, _tracking_weight

    d = tmp_projects_root / "bare"
    d.mkdir(parents=True, exist_ok=True)
    assert _episode_horizon_s(d) is None
    assert _tracking_weight(d) == 1.0


def test_a_world_project_declares_horizon_and_demotes_the_clip(
    tmp_projects_root: Path,
) -> None:
    """The horizon and imitation balance come from the same selection."""
    from backend.routes.references import (WORLD_TRACKING_WEIGHT,
                                           _episode_horizon_s,
                                           _tracking_weight)

    d = tmp_projects_root / "world"
    _write_selection(d)
    assert _episode_horizon_s(d) == 20.0
    assert _tracking_weight(d) == WORLD_TRACKING_WEIGHT


def test_changed_task_content_invalidates_same_clip_mode_reward(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """A selection file can keep the same ref paths while the immutable
    objects behind them change. The binding hashes the referenced content, not
    merely the path or clip id."""
    _write_composite(refs_root)
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_selection(project_dir, episode_s=20.0)
    body = _scaffold(client, slug)

    task_path = project_dir / "env" / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["shared"]["termination"]["episode_length_s"] = 12.0
    task_path.write_text(json.dumps(task), encoding="utf-8")

    listing = client.get(f"/projects/{slug}/mode-rewards").json()
    mine = next(
        item for item in listing["mode_rewards"]
        if item["filename"] == body["filename"]
    )
    assert mine["context_current"] is False
    refused = client.post(
        PROMOTE_URL.format(slug=slug),
        json={"filename": body["filename"], "allow_unauthored": True},
    )
    assert refused.status_code == 409, refused.text
    assert "clip-id equality alone" in refused.json()["detail"]


def test_a_selection_without_a_goal_is_not_a_mission(
    tmp_projects_root: Path,
) -> None:
    """A world with no goal has nothing to make progress towards, so the clip
    stays the objective and the brief stays empty."""
    from backend.routes.references import (_mission_brief, _selection_specs,
                                           _tracking_weight)

    d = tmp_projects_root / "goalless"
    _write_selection(d, with_goal=False)
    assert _mission_brief(*_selection_specs(d)) == ""
    assert _tracking_weight(d) == 1.0


def test_the_brief_offers_only_the_channel_a_reward_may_read(
    tmp_projects_root: Path,
) -> None:
    """`waypoint_index` is metric_only and absent from the contract's info
    keys; a term reading it is rejected as ungrounded after the model call."""
    from backend.routes.references import _mission_brief, _selection_specs

    d = tmp_projects_root / "channels"
    _write_selection(d)
    brief = _mission_brief(*_selection_specs(d))
    may, metric_only = brief.split("METRICS ONLY", 1)
    assert "goal__complete_course__waypoint_distance" in may
    assert "goal__complete_course__waypoint_index" in metric_only
    assert "platform box_01" in brief


def test_a_scaffold_on_a_world_project_carries_both_fixes(
    client: TestClient, refs_root: Path, tmp_projects_root: Path,
) -> None:
    """The route preserves certified cadence and prices the clip as a prior."""
    from backend.routes.references import WORLD_TRACKING_WEIGHT

    _write_composite(refs_root)
    slug = _make_project(client)
    _write_selection(tmp_projects_root / slug)
    body = _scaffold(client, slug)

    src = (tmp_projects_root / slug / "rewards" / body["filename"]).read_text()
    from sculptor.mode_rewards import reward_spec_from_source

    spec = reward_spec_from_source(src)
    assert f"TRACKING_W = {WORLD_TRACKING_WEIGHT!r}" in src
    assert spec["episode_horizon_s"] == 20.0
    assert spec["clip_time_scale"] == 1.0
    # A sampled clip's wall-clock span is (N - 1) / fps: 240 samples at
    # 120 Hz span 1.991666… s, leaving the remainder as terminal hold.
    assert spec["terminal_hold_s"] == pytest.approx(20.0 - (239.0 / 120.0))
    assert spec["schedule_policy"] == (
        "certified_clip_cadence_then_terminal_hold"
    )
    assert spec["mode_binding"]["clip_id"] == "novel-jump-kick--g1"
    assert spec["mode_binding"]["robot"] == "g1"
    assert len(spec["mode_binding"]["clip_sha256"]) == 64
    assert len(spec["mode_binding"]["graph_sha256"]) == 64
    assert len(spec["mode_binding"]["execution_manifest_digest"]) == 64
    assert "REFERENCE_DURATION_S = 1.9916666666666667" in src


def test_promotion_records_which_file_it_came_from(
    client: TestClient, refs_root: Path, tmp_projects_root: Path, monkeypatch,
) -> None:
    """`promote` rewrites REWARD_SPEC, so the copy never digests equal to its
    source. Without the source digest recorded, "is what trains still what I
    authored?" is unanswerable — and the panel answered "yes" unconditionally,
    which disabled the promote button forever after the first promotion."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    final = _author_all(client, slug, monkeypatch)
    client.post(PROMOTE_URL.format(slug=slug), json={"filename": final})

    body = client.get(f"/projects/{slug}/mode-rewards").json()
    mine = next(f for f in body["mode_rewards"] if f["filename"] == final)

    assert body["promoted"]["source_filename"] == final
    assert body["promoted"]["source_sha256"] == mine["digest"]


def test_re_authoring_after_a_promotion_reads_as_not_yet_training(
    client: TestClient, refs_root: Path, tmp_projects_root: Path, monkeypatch,
) -> None:
    """The state the user was stuck in: everything re-authored, and the only
    control that could make it train was disabled behind a stale version."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_composite(refs_root)
    slug = _make_project(client)
    _scaffold(client, slug)
    final = _author_all(client, slug, monkeypatch)
    client.post(PROMOTE_URL.format(slug=slug), json={"filename": final})

    # Re-scaffold in place: same filename, different bytes.
    _scaffold(client, slug, overwrite=True)
    body = client.get(f"/projects/{slug}/mode-rewards").json()

    # The listing is newest-first, and the panel adopts the first match — so
    # this is the file the user is looking at.
    current = body["mode_rewards"][0]
    assert body["promoted"] is not None, "v<n>.py is still what trains"
    assert current["digest"] != body["promoted"]["source_sha256"], (
        "the file on screen is not the one training, and the panel has to be "
        "able to say so")
    # The earlier chained file the promotion DID come from is still on disk;
    # matching by filename alone would therefore have said "this is training".
    assert any(f["digest"] == body["promoted"]["source_sha256"]
               for f in body["mode_rewards"])
