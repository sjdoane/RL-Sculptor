"""tests/test_archive.py — offline tests for the durable mission archive.

Builds a synthetic `.missions/<slug>/` tree on disk (no real training run),
then exercises `archive_mission`'s retention rules: checkpoint keep-set
(final ∪ best ∪ pinned), checkpoint-adjacent trajectory/keyframes riding
along ONLY with kept checkpoints, `logs/`/`edit_candidates/` never copied,
manifest byte accounting, and incremental-call idempotency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sculptor.archive import (
    ArchiveError,
    archive_mission,
    list_saved,
    read_manifest,
    saved_root,
)


# ── fixture writer ───────────────────────────────────────────────────────
def _write_iter(
    stage_dir: Path, idx: int, *, fitness: float,
    with_checkpoint: bool = True, ckpt_bytes: bytes = b"ckpt-payload",
) -> Path:
    iter_dir = stage_dir / "runs" / f"iter_{idx}"
    rollout_dir = iter_dir / "rollout"
    rollout_dir.mkdir(parents=True)

    (iter_dir / "metrics.json").write_text(
        json.dumps({"metrics": {"mean_return": fitness}}), encoding="utf-8")
    (iter_dir / "diagnosis.json").write_text("{}", encoding="utf-8")
    (iter_dir / "reward_spec.json").write_text("{}", encoding="utf-8")
    (iter_dir / "realism_audit.json").write_text("{}", encoding="utf-8")
    (iter_dir / "partition_gate.json").write_text("{}", encoding="utf-8")
    (iter_dir / "env_spec.json").write_text("{}", encoding="utf-8")
    (iter_dir / "edit_candidates.json").write_text("{}", encoding="utf-8")
    (iter_dir / "reward_trajectory.json").write_text("{}", encoding="utf-8")

    (rollout_dir / "rollout.mp4").write_bytes(b"x" * 4096)
    (rollout_dir / "behavior.json").write_text(
        json.dumps({"fitness": fitness}), encoding="utf-8")
    (rollout_dir / "mjcf_limits.json").write_text("{}", encoding="utf-8")
    (rollout_dir / "reward_trajectory.json").write_text("{}", encoding="utf-8")
    (rollout_dir / "trajectory.npz").write_bytes(b"npz-payload")
    keyframes = rollout_dir / "keyframes"
    keyframes.mkdir()
    (keyframes / "frame_00.png").write_bytes(b"png-bytes")

    # NEVER-copy dirs — must survive untouched in the source but never
    # show up in the archived copy.
    logs_dir = iter_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "big.log").write_bytes(b"z" * 1_000_000)
    ec_dir = iter_dir / "edit_candidates"
    ec_dir.mkdir()
    (ec_dir / "candidate_0.py").write_text("# raw llm sample\n", encoding="utf-8")

    if with_checkpoint:
        (iter_dir / "checkpoint.pt").write_bytes(ckpt_bytes)

    return iter_dir


def _write_stage(
    mission_dir: Path, name: str, *, n_iters: int, fitnesses: list[float],
    status: str = "succeeded",
) -> Path:
    stage_dir = mission_dir / "stages" / name
    (stage_dir / "rewards").mkdir(parents=True)
    (stage_dir / "rewards" / "v0.py").write_text("REWARD_SPEC = {}\n", encoding="utf-8")
    (stage_dir / "rewards" / "__pycache__").mkdir()
    (stage_dir / "rewards" / "__pycache__" / "v0.cpython-313.pyc").write_bytes(b"bytecode")
    (stage_dir / "env").mkdir()
    (stage_dir / "env" / "current.json").write_text("{}", encoding="utf-8")
    (stage_dir / "reports").mkdir()
    (stage_dir / "reports" / "metric_history.json").write_text(
        json.dumps({"primary_metric": "mean_return", "history": fitnesses}),
        encoding="utf-8")
    (stage_dir / "config.toml").write_text("[target]\nname = \"t\"\n", encoding="utf-8")
    (stage_dir / "CHANGELOG.md").write_text("# changelog\n", encoding="utf-8")
    (stage_dir / "kg_seeds.yml").write_text("seeds: []\n", encoding="utf-8")

    for i in range(n_iters):
        _write_iter(stage_dir, i, fitness=fitnesses[i])

    return stage_dir


def _write_mission(
    tmp_path: Path, *, mission_slug: str = "jump-mission",
    stages: dict[str, dict] | None = None,
) -> Path:
    mission_dir = tmp_path / "project" / ".missions" / mission_slug
    mission_dir.mkdir(parents=True)
    stages = stages or {
        "launch": {"n_iters": 3, "fitnesses": [1.0, 5.0, 3.0]},
    }
    stage_docs = []
    for name, spec in stages.items():
        _write_stage(
            mission_dir, name, n_iters=spec["n_iters"],
            fitnesses=spec["fitnesses"], status=spec.get("status", "succeeded"))
        stage_docs.append({"name": name, "status": spec.get("status", "succeeded")})

    (mission_dir / "mission.json").write_text(json.dumps({
        "schema_version": 1,
        "goal": "jump really high",
        "decomposition_model": "stub",
        "decomposition_rationale": "stub",
        "stages": stage_docs,
    }), encoding="utf-8")
    (mission_dir / "telemetry.json").write_text('{"n_stages": 1}', encoding="utf-8")
    (mission_dir / "provenance.json").write_text('{"stages": []}', encoding="utf-8")
    (mission_dir / "llm_calls.jsonl").write_text('{"call": 1}\n', encoding="utf-8")
    (mission_dir / "stage_metrics" / "launch").mkdir(parents=True)
    (mission_dir / "stage_metrics" / "launch" / "metric.py").write_text(
        "def score(t): return 0.0\n", encoding="utf-8")
    (mission_dir / "stage_metrics" / "launch" / "__pycache__").mkdir()
    (mission_dir / "stage_metrics" / "launch" / "__pycache__" / "x.pyc").write_bytes(b"z")

    return mission_dir


# ── saved_root ───────────────────────────────────────────────────────────
def test_saved_root_default(monkeypatch):
    monkeypatch.delenv("RS_SAVED_ROOT", raising=False)
    root = saved_root()
    assert root == Path.home() / ".local" / "share" / "reward-sculptor" / "saved"


def test_saved_root_env_override(monkeypatch, tmp_path):
    override = tmp_path / "custom_saved"
    monkeypatch.setenv("RS_SAVED_ROOT", str(override))
    assert saved_root() == override


# ── archive_mission: basic structure ─────────────────────────────────────
def test_archive_mission_raises_without_mission_json(tmp_path):
    not_a_mission = tmp_path / "not_a_mission"
    not_a_mission.mkdir()
    with pytest.raises(ArchiveError):
        archive_mission(not_a_mission, tmp_path / "dest", project_slug="proj")


def test_archive_mission_copies_mission_root_artifacts(tmp_path):
    mission_dir = _write_mission(tmp_path)
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    assert result.entry_dir.is_dir()
    assert result.entry_dir.parent == dest_root
    assert (result.entry_dir / "mission.json").is_file()
    assert (result.entry_dir / "telemetry.json").is_file()
    assert (result.entry_dir / "provenance.json").is_file()
    assert (result.entry_dir / "llm_calls.jsonl").is_file()
    assert (result.entry_dir / "stage_metrics" / "launch" / "metric.py").is_file()
    # __pycache__ under stage_metrics must not survive.
    assert not (result.entry_dir / "stage_metrics" / "launch" / "__pycache__").exists()


def test_entry_id_format(tmp_path):
    mission_dir = _write_mission(tmp_path, mission_slug="jump-mission")
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="myproj")
    parts = result.entry_dir.name.split("--")
    assert len(parts) == 3
    assert parts[0] == "myproj"
    assert parts[1] == "jump-mission"
    # UTC stamp: YYYYMMDDTHHMMSSZ
    assert len(parts[2]) == 16
    assert parts[2].endswith("Z")
    assert parts[2][8] == "T"


# ── checkpoint retention: final ∪ best ∪ pinned ──────────────────────────
def test_checkpoint_retention_final_and_best(tmp_path):
    # 3 iters, fitness makes iter 1 the best; iter 2 is final.
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 3, "fitnesses": [1.0, 5.0, 3.0]}})
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    kept_iters = {kc["iter"] for kc in result.kept_checkpoints}
    assert kept_iters == {1, 2}

    reasons_by_iter = {kc["iter"]: kc["reason"] for kc in result.kept_checkpoints}
    assert reasons_by_iter[1] == "best"
    assert reasons_by_iter[2] == "final"

    launch_dir = result.entry_dir / "stages" / "launch" / "runs"
    assert (launch_dir / "iter_1" / "checkpoint.pt").is_file()
    assert (launch_dir / "iter_2" / "checkpoint.pt").is_file()
    # iter 0 (neither final nor best) must NOT carry a checkpoint.
    assert not (launch_dir / "iter_0" / "checkpoint.pt").exists()


def test_checkpoint_retention_with_pinned(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 3, "fitnesses": [1.0, 5.0, 3.0]}})
    dest_root = tmp_path / "saved"
    result = archive_mission(
        mission_dir, dest_root, project_slug="proj",
        pinned={"launch": {0}})

    kept_iters = {kc["iter"] for kc in result.kept_checkpoints}
    assert kept_iters == {0, 1, 2}
    reasons_by_iter = {kc["iter"]: kc["reason"] for kc in result.kept_checkpoints}
    assert reasons_by_iter[0] == "pinned"

    launch_dir = result.entry_dir / "stages" / "launch" / "runs"
    assert (launch_dir / "iter_0" / "checkpoint.pt").is_file()


def test_final_and_best_same_iter_reports_once(tmp_path):
    # Monotonically increasing fitness → final IS the best. Must not
    # appear twice in kept_checkpoints.
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 3, "fitnesses": [1.0, 2.0, 5.0]}})
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    iters = [kc["iter"] for kc in result.kept_checkpoints]
    assert iters == [2]
    assert result.kept_checkpoints[0]["reason"] == "final"


# ── checkpoint-adjacent files ride ONLY with kept checkpoints ────────────
def test_trajectory_and_keyframes_only_beside_kept_checkpoints(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 3, "fitnesses": [1.0, 5.0, 3.0]}})
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    runs = result.entry_dir / "stages" / "launch" / "runs"
    # Kept (iter 1 = best, iter 2 = final): trajectory.npz + keyframes/.
    assert (runs / "iter_1" / "rollout" / "trajectory.npz").is_file()
    assert (runs / "iter_1" / "rollout" / "keyframes" / "frame_00.png").is_file()
    assert (runs / "iter_2" / "rollout" / "trajectory.npz").is_file()
    assert (runs / "iter_2" / "rollout" / "keyframes" / "frame_00.png").is_file()
    # Dropped (iter 0): no trajectory.npz, no keyframes/.
    assert not (runs / "iter_0" / "rollout" / "trajectory.npz").exists()
    assert not (runs / "iter_0" / "rollout" / "keyframes").exists()
    # But iter 0's lightweight artifacts DID survive.
    assert (runs / "iter_0" / "metrics.json").is_file()
    assert (runs / "iter_0" / "rollout" / "rollout.mp4").is_file()
    assert (runs / "iter_0" / "rollout" / "behavior.json").is_file()


# ── never-copy dirs ───────────────────────────────────────────────────────
def test_logs_and_edit_candidates_never_copied(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 3, "fitnesses": [1.0, 5.0, 3.0]}})
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    runs = result.entry_dir / "stages" / "launch" / "runs"
    for i in range(3):
        assert not (runs / f"iter_{i}" / "logs").exists()
        assert not (runs / f"iter_{i}" / "edit_candidates").exists()
    # But edit_candidates.json (the FILE, distinct from the dir) is kept.
    assert (runs / "iter_0" / "edit_candidates.json").is_file()


def test_rollout_fresh_dirs_only_behavior_json(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 1, "fitnesses": [1.0]}})
    iter_dir = mission_dir / "stages" / "launch" / "runs" / "iter_0"
    fresh_dir = iter_dir / "rollout_fresh_0"
    fresh_dir.mkdir()
    (fresh_dir / "behavior.json").write_text('{"fitness": 2.0}', encoding="utf-8")
    (fresh_dir / "rollout.mp4").write_bytes(b"y" * 4096)
    (fresh_dir / "trajectory.npz").write_bytes(b"npz")

    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    dst_fresh = (result.entry_dir / "stages" / "launch" / "runs" / "iter_0"
                 / "rollout_fresh_0")
    assert (dst_fresh / "behavior.json").is_file()
    assert not (dst_fresh / "rollout.mp4").exists()
    assert not (dst_fresh / "trajectory.npz").exists()


# ── manifest byte accounting ──────────────────────────────────────────────
def test_manifest_total_and_dropped_bytes(tmp_path):
    ckpt_payload = b"K" * 10_000
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 3, "fitnesses": [1.0, 5.0, 3.0]}})
    # Overwrite checkpoints with a known size so the math is exact.
    for i in range(3):
        (mission_dir / "stages" / "launch" / "runs" / f"iter_{i}"
         / "checkpoint.pt").write_bytes(ckpt_payload)

    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    manifest = read_manifest(result.entry_dir)
    assert manifest["total_bytes"] == result.total_bytes
    assert manifest["dropped_bytes"] == result.dropped_bytes
    # iter 0's checkpoint (10_000 bytes) is the only dropped checkpoint;
    # trajectory.npz + the keyframes/ PNG are dropped alongside it.
    npz_len = len(b"npz-payload")
    png_len = len(b"png-bytes")
    assert result.dropped_bytes == 10_000 + npz_len + png_len
    assert result.total_bytes > 0

    # Every byte actually on disk under entry_dir must be <= total_bytes
    # (manifest.json itself isn't counted, so this is a `<=`, not `==`).
    on_disk = sum(
        p.stat().st_size for p in result.entry_dir.rglob("*") if p.is_file())
    assert on_disk >= result.total_bytes  # manifest.json + tracked files


def test_manifest_stage_shape(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 3, "fitnesses": [1.0, 5.0, 3.0]}})
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    manifest = read_manifest(result.entry_dir)
    assert manifest["schema"] == 1
    assert manifest["project_slug"] == "proj"
    assert manifest["mission_slug"] == "jump-mission"
    assert manifest["goal"] == "jump really high"
    assert len(manifest["stages"]) == 1
    stage = manifest["stages"][0]
    assert stage["name"] == "launch"
    assert stage["status"] == "succeeded"
    assert stage["best_metric"] == pytest.approx(5.0)
    assert stage["n_iters"] == 3
    assert len(stage["videos"]) == 3
    kept_iters = {kc["iter"] for kc in stage["kept_checkpoints"]}
    assert kept_iters == {1, 2}


# ── incremental idempotency ────────────────────────────────────────────────
def test_incremental_second_call_reuses_entry_dir(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 2, "fitnesses": [1.0, 2.0]}})
    dest_root = tmp_path / "saved"

    result1 = archive_mission(mission_dir, dest_root, project_slug="proj")
    # Simulate a later stage-completion event on the SAME mission run —
    # more iterations have landed by now.
    _write_iter(mission_dir / "stages" / "launch", 2, fitness=10.0)
    (mission_dir / "stages" / "launch" / "reports" / "metric_history.json").write_text(
        json.dumps({"primary_metric": "mean_return",
                    "history": [1.0, 2.0, 10.0]}), encoding="utf-8")
    result2 = archive_mission(mission_dir, dest_root, project_slug="proj")

    assert result1.entry_dir == result2.entry_dir
    # Only ONE entry dir should exist under dest_root.
    entries = [d for d in dest_root.iterdir() if d.is_dir()]
    assert len(entries) == 1

    runs = result2.entry_dir / "stages" / "launch" / "runs"
    assert (runs / "iter_2").is_dir()  # newly-appeared iter picked up
    kept = {kc["iter"] for kc in result2.kept_checkpoints}
    assert kept == {2}  # new iter is both final AND best now


def test_incremental_call_does_not_duplicate_files(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 2, "fitnesses": [1.0, 2.0]}})
    dest_root = tmp_path / "saved"

    archive_mission(mission_dir, dest_root, project_slug="proj")
    result2 = archive_mission(mission_dir, dest_root, project_slug="proj")

    # Re-running with no new data must not change kept-checkpoint counts
    # or blow up totals (copytree merge is idempotent for identical files).
    kept = {kc["iter"] for kc in result2.kept_checkpoints}
    assert kept == {1}  # final == best for [1.0, 2.0]
    runs = result2.entry_dir / "stages" / "launch" / "runs"
    assert sum(1 for _ in runs.iterdir()) == 2


def test_incremental_updates_status_in_manifest(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 1, "fitnesses": [1.0], "status": "training"}})
    dest_root = tmp_path / "saved"
    result1 = archive_mission(mission_dir, dest_root, project_slug="proj")
    manifest1 = read_manifest(result1.entry_dir)
    assert manifest1["stages"][0]["status"] == "training"

    # Mission orchestrator flips the stage to succeeded and re-archives.
    mission_doc = json.loads((mission_dir / "mission.json").read_text(encoding="utf-8"))
    mission_doc["stages"][0]["status"] = "succeeded"
    (mission_dir / "mission.json").write_text(
        json.dumps(mission_doc), encoding="utf-8")

    result2 = archive_mission(mission_dir, dest_root, project_slug="proj")
    manifest2 = read_manifest(result2.entry_dir)
    assert manifest2["stages"][0]["status"] == "succeeded"
    assert result1.entry_dir == result2.entry_dir


def test_non_incremental_forces_new_entry(tmp_path, monkeypatch):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 1, "fitnesses": [1.0]}})
    dest_root = tmp_path / "saved"

    # The stamp has whole-second resolution (by design — see the module
    # docstring's YYYYMMDDTHHMMSSZ format), so two real-clock calls in
    # the same test can collide. Force distinct seconds deterministically
    # rather than relying on wall-clock luck / a real sleep.
    import sculptor.archive as archive_mod

    class _FrozenDatetime(archive_mod.datetime):
        _now = archive_mod.datetime(2026, 1, 1, 0, 0, 0, tzinfo=archive_mod.timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls._now

    monkeypatch.setattr(archive_mod, "datetime", _FrozenDatetime)

    result1 = archive_mission(
        mission_dir, dest_root, project_slug="proj", incremental=True)
    _FrozenDatetime._now = _FrozenDatetime._now.replace(second=1)
    result2 = archive_mission(
        mission_dir, dest_root, project_slug="proj", incremental=False)

    assert result1.entry_dir != result2.entry_dir
    entries = [d for d in dest_root.iterdir() if d.is_dir()]
    assert len(entries) == 2


# ── 2-stage variant ────────────────────────────────────────────────────────
def test_two_stage_mission_each_gets_own_retention(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={
            "crouch": {"n_iters": 2, "fitnesses": [1.0, 4.0]},
            "launch": {"n_iters": 3, "fitnesses": [2.0, 8.0, 6.0]},
        })
    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    by_stage: dict[str, set[int]] = {}
    for kc in result.kept_checkpoints:
        by_stage.setdefault(kc["stage"], set()).add(kc["iter"])
    # crouch: final==best==iter 1.
    assert by_stage["crouch"] == {1}
    # launch: best=iter1(8.0), final=iter2.
    assert by_stage["launch"] == {1, 2}

    manifest = read_manifest(result.entry_dir)
    assert {s["name"] for s in manifest["stages"]} == {"crouch", "launch"}
    for stage_dir_name in ("crouch", "launch"):
        assert (result.entry_dir / "stages" / stage_dir_name
                / "config.toml").is_file()


# ── list_saved ───────────────────────────────────────────────────────────
def test_list_saved_empty_dir(tmp_path):
    assert list_saved(tmp_path / "does_not_exist") == []


def test_list_saved_returns_manifests(tmp_path):
    mission_dir = _write_mission(tmp_path, mission_slug="m1")
    dest_root = tmp_path / "saved"
    archive_mission(mission_dir, dest_root, project_slug="proj")

    entries = list_saved(dest_root)
    assert len(entries) == 1
    assert entries[0]["mission_slug"] == "m1"


def test_list_saved_skips_corrupt_entries(tmp_path):
    dest_root = tmp_path / "saved"
    dest_root.mkdir()
    bad = dest_root / "corrupt-entry"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not valid json", encoding="utf-8")

    mission_dir = _write_mission(tmp_path, mission_slug="m2")
    archive_mission(mission_dir, dest_root, project_slug="proj")

    entries = list_saved(dest_root)
    assert len(entries) == 1
    assert entries[0]["mission_slug"] == "m2"


# ── checkpoint-less stage (no checkpoints written at all) ─────────────────
def test_stage_with_no_checkpoints_keeps_nothing_but_still_copies(tmp_path):
    mission_dir = _write_mission(
        tmp_path,
        stages={"launch": {"n_iters": 2, "fitnesses": [1.0, 2.0]}})
    for i in range(2):
        ckpt = (mission_dir / "stages" / "launch" / "runs" / f"iter_{i}"
                / "checkpoint.pt")
        ckpt.unlink()

    dest_root = tmp_path / "saved"
    result = archive_mission(mission_dir, dest_root, project_slug="proj")

    assert result.kept_checkpoints == []
    runs = result.entry_dir / "stages" / "launch" / "runs"
    # Lightweight artifacts still copied.
    assert (runs / "iter_0" / "metrics.json").is_file()
    assert (runs / "iter_1" / "rollout" / "rollout.mp4").is_file()
