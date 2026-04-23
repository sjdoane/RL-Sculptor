"""Tests for the live-clip streamer + /clips + /frames WS.

Avoids running a real sculpt subprocess — we drive the streamer
directly with a synthetic job and a fake source mp4. The R3 saturation
test uses `set_render_slowdown` to make two overlapping renders race.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.services.job_manager import Job, JobManager
from backend.services.rollout_streamer import (
    RolloutStreamer,
    set_render_slowdown,
)


def _prep_source_mp4(project_dir: Path, iter_n: int) -> Path:
    """Write a tiny valid mp4 at runs/iter_<n>/rollout/rollout.mp4 so
    the streamer has something real to truncate."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = project_dir / "runs" / f"iter_{iter_n}" / "rollout" / "rollout.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 3 seconds of a solid-colour test pattern at 30fps.
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=darkred:s=160x120:d=3:r=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True, capture_output=True,
    )
    assert out.stat().st_size > 2048
    return out


def _fake_job(slug: str, run_id: str = "job_fake") -> Job:
    return Job(
        job_id=run_id, kind="sculpt_run", project_slug=slug, status="running"
    )


# ── happy path ────────────────────────────────────────────────────────
def test_streamer_emits_new_clip(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    _prep_source_mp4(project_dir, iter_n=0)

    job = _fake_job("proj")

    async def _run():
        streamer = RolloutStreamer()
        await streamer.maybe_render(job, 0, project_dir)
        # Give the background task a moment to finish the ffmpeg
        # subprocess — bounded at 30 s in production, tiny here.
        for _ in range(100):
            if any(e.get("type") == "new_clip" for e in job.events):
                break
            await asyncio.sleep(0.05)

    asyncio.run(_run())
    clips = [e for e in job.events if e.get("type") == "new_clip"]
    assert len(clips) == 1, job.events
    ev = clips[0]
    assert ev["iter"] == 0
    assert ev["url"].endswith("/clips/iter_0.mp4")
    clip_path = project_dir / "uploads" / "live_clips" / "job_fake" / "iter_0.mp4"
    assert clip_path.is_file()
    assert clip_path.stat().st_size > 128


# ── R3: saturation → clip_skipped ─────────────────────────────────────
def test_streamer_skips_when_saturated(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    for n in range(3):
        _prep_source_mp4(project_dir, iter_n=n)

    job = _fake_job("proj")
    set_render_slowdown(0.4)

    async def _run():
        streamer = RolloutStreamer(max_concurrent=1)
        # Fire three renders in quick succession. With max_concurrent=1
        # and the slowdown, renders 2 and 3 must be skipped (NOT
        # queued) per R3.
        await streamer.maybe_render(job, 0, project_dir)
        await streamer.maybe_render(job, 1, project_dir)
        await streamer.maybe_render(job, 2, project_dir)
        # Wait for the first render to finish.
        for _ in range(200):
            new_clip_count = sum(
                1 for e in job.events if e.get("type") == "new_clip"
            )
            if new_clip_count >= 1:
                break
            await asyncio.sleep(0.05)

    try:
        asyncio.run(_run())
    finally:
        set_render_slowdown(None)

    new_clips = [e for e in job.events if e.get("type") == "new_clip"]
    skipped = [e for e in job.events if e.get("type") == "clip_skipped"]
    assert len(new_clips) == 1, (new_clips, skipped)
    assert len(skipped) == 2, skipped
    for sk in skipped:
        assert sk["reason"] == "render_pool_saturated"


def test_streamer_skips_when_source_missing(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    job = _fake_job("proj")

    async def _run():
        streamer = RolloutStreamer()
        await streamer.maybe_render(job, 0, project_dir)
        for _ in range(40):
            if any(e.get("type") in ("clip_skipped", "new_clip") for e in job.events):
                break
            await asyncio.sleep(0.05)

    asyncio.run(_run())
    skipped = [e for e in job.events if e.get("type") == "clip_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "source_missing_or_empty"


# ── routes + WS ───────────────────────────────────────────────────────
def _make_project_with_library(client: TestClient, name: str) -> str:
    r = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 3, "behavior_goal": "hop"},
    )
    slug = r.json()["slug"]
    r = client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "hopper"},
    )
    assert r.status_code == 200
    return slug


def test_clip_file_route_serves_mp4(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project_with_library(client, "ClipFile")
    # Inject a synthetic run + a clip file on disk.
    app = client.app  # type: ignore[attr-defined]
    jm: JobManager = app.state.job_manager
    run_id = "job_clip_serve"
    jm._jobs[run_id] = Job(
        job_id=run_id, kind="sculpt_run", project_slug=slug, status="completed",
    )
    project_dir = tmp_projects_root / slug
    _prep_source_mp4(project_dir, iter_n=0)
    clip_dir = project_dir / "uploads" / "live_clips" / run_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    # Use the source itself as the "clip" for file-serving smoke.
    shutil.copyfile(
        project_dir / "runs" / "iter_0" / "rollout" / "rollout.mp4",
        clip_dir / "iter_0.mp4",
    )

    r = client.get(f"/projects/{slug}/runs/{run_id}/clips/iter_0.mp4")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 512

    # Path traversal refused.
    rbad = client.get(f"/projects/{slug}/runs/{run_id}/clips/..%2Fbreak.mp4")
    assert rbad.status_code in (404, 400)


def test_iter_rollout_route_serves_mp4(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project_with_library(client, "IterRollout")
    app = client.app  # type: ignore[attr-defined]
    jm: JobManager = app.state.job_manager
    run_id = "job_iter_serve"
    jm._jobs[run_id] = Job(
        job_id=run_id, kind="sculpt_run", project_slug=slug, status="completed",
    )
    project_dir = tmp_projects_root / slug
    _prep_source_mp4(project_dir, iter_n=2)

    r = client.get(f"/projects/{slug}/runs/{run_id}/iterations/2/rollout")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    r404 = client.get(f"/projects/{slug}/runs/{run_id}/iterations/42/rollout")
    assert r404.status_code == 404


def test_frames_ws_replays_clips(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project_with_library(client, "FramesWs")
    app = client.app  # type: ignore[attr-defined]
    jm: JobManager = app.state.job_manager
    run_id = "job_ws_clips"
    job = Job(
        job_id=run_id, kind="sculpt_run", project_slug=slug, status="completed",
    )
    # Pre-seed the event stream with two clips.
    job.emit({
        "type": "new_clip", "iter": 0,
        "url": f"/api/projects/{slug}/runs/{run_id}/clips/iter_0.mp4",
        "duration_s": 2.0,
    })
    job.emit({
        "type": "clip_skipped", "iter": 1, "reason": "source_missing_or_empty",
    })
    job.emit({
        "type": "new_clip", "iter": 2,
        "url": f"/api/projects/{slug}/runs/{run_id}/clips/iter_2.mp4",
        "duration_s": 2.0,
    })
    jm._jobs[run_id] = job

    with client.websocket_connect(
        f"/ws/projects/{slug}/runs/{run_id}/frames"
    ) as ws:
        seen = []
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg["type"] == "terminal":
                break
    types = [m["type"] for m in seen]
    assert types[0] == "connected"
    assert types.count("new_clip") == 2
    assert types.count("clip_skipped") == 1
    assert "terminal" in types
