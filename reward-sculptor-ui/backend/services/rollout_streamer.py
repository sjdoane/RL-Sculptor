"""Per-run live-clip generator.

On every filesystem-derived `rollout_done` event the streamer's
`maybe_render(job, iter_n, project_dir)` method:

  1. confirms the sculptor's full 20 s rollout.mp4 is on disk + non-empty;
  2. gates on a bounded in-flight counter (per Prompt 9 R3 — we NEVER
     block the training subprocess on render backpressure; if the pool
     is saturated we emit `clip_skipped` and return immediately);
  3. schedules ffmpeg-subprocess truncation to 2 s via
     `asyncio.to_thread`; the work is pure CPU (R1);
  4. on success writes the clip to
     `<project>/uploads/live_clips/<run_id>/iter_<i>.mp4` and emits
     `new_clip` on the job's event stream so the `/frames` WS clients
     see it immediately.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from backend.services.job_manager import Job


log = logging.getLogger("reward-sculptor-ui.rollout_streamer")

MAX_CONCURRENT_RENDERS = 2
CLIP_DURATION_S = 2.0
MIN_SOURCE_BYTES = 2048
# Set to a float to test R3's saturation behavior from a unit test.
# None in production.
_RENDER_SLOWDOWN_S: Optional[float] = None


class RolloutStreamer:
    def __init__(
        self,
        *,
        max_concurrent: int = MAX_CONCURRENT_RENDERS,
        clip_duration_s: float = CLIP_DURATION_S,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.clip_duration_s = clip_duration_s
        self._in_flight: set[int] = set()
        self._completed: set[int] = set()
        self._lock = asyncio.Lock()

    def clip_path(
        self, project_dir: Path, run_id: str, iter_n: int
    ) -> Path:
        return (
            project_dir
            / "uploads"
            / "live_clips"
            / run_id
            / f"iter_{iter_n}.mp4"
        )

    async def maybe_render(
        self,
        job: Job,
        iter_n: int,
        project_dir: Path,
    ) -> None:
        """Fire-and-forget: schedule a clip render for this iter if
        capacity allows. Returns immediately — work runs in the
        background. NEVER awaits the render itself from the caller's
        frame."""
        async with self._lock:
            if iter_n in self._in_flight or iter_n in self._completed:
                return
            if len(self._in_flight) >= self.max_concurrent:
                # R3: saturated — skip the clip, don't queue.
                log.warning(
                    "live-clip pool saturated (%d in flight) — skipping iter %d",
                    len(self._in_flight),
                    iter_n,
                )
                job.emit({
                    "type": "clip_skipped",
                    "iter": iter_n,
                    "reason": "render_pool_saturated",
                    "in_flight": len(self._in_flight),
                })
                return
            self._in_flight.add(iter_n)
        # Release the lock before spawning the task so new incoming
        # iter_completed events can be checked against the (now-updated)
        # in_flight count without waiting for this render to finish.
        asyncio.create_task(self._render_one(job, iter_n, project_dir))

    async def _render_one(
        self, job: Job, iter_n: int, project_dir: Path
    ) -> None:
        try:
            run_id = job.job_id
            source = (
                project_dir / "runs" / f"iter_{iter_n}" / "rollout" / "rollout.mp4"
            )
            if not source.is_file() or source.stat().st_size < MIN_SOURCE_BYTES:
                job.emit({
                    "type": "clip_skipped",
                    "iter": iter_n,
                    "reason": "source_missing_or_empty",
                })
                return
            out = self.clip_path(project_dir, run_id, iter_n)
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            ok, stderr = await asyncio.to_thread(
                _truncate_mp4, source, out, self.clip_duration_s
            )
            elapsed = time.time() - t0
            if not ok:
                job.emit({
                    "type": "clip_skipped",
                    "iter": iter_n,
                    "reason": "ffmpeg_failed",
                    "stderr_tail": stderr[-400:] if stderr else "",
                })
                return
            job.emit({
                "type": "new_clip",
                "iter": iter_n,
                "url": f"/api/projects/{job.project_slug}/runs/{run_id}/clips/iter_{iter_n}.mp4",
                "duration_s": self.clip_duration_s,
                "render_elapsed_s": round(elapsed, 3),
                "size_bytes": out.stat().st_size if out.is_file() else 0,
            })
        finally:
            async with self._lock:
                self._in_flight.discard(iter_n)
                self._completed.add(iter_n)


# ── ffmpeg truncation (CPU) ───────────────────────────────────────────
def _truncate_mp4(
    source: Path, dest: Path, duration_s: float
) -> tuple[bool, str]:
    """Cut the first `duration_s` of `source` into `dest`.

    Re-encodes with libx264 ultrafast — stream-copy (`-c copy`) is
    faster but fails on sources whose first `duration_s` s don't
    contain a keyframe, which is common for the adapter's rollout.mp4.
    Ultrafast runs in ~50-150 ms per 2 s clip on CPU (Prompt 9 R1 + R3).

    Retry loop: the filesystem watcher can fire while the adapter is
    still flushing its mp4 to disk ("moov atom not found" — the moov
    atom is written last on file close). We retry up to 6 times with
    exponential backoff up to ~3 s total; if still failing, we surface
    a clip_skipped event rather than block forever.
    """
    if _RENDER_SLOWDOWN_S is not None:
        # R3 test harness: artificial per-render sleep so saturation is
        # triggerable from unit tests.
        time.sleep(_RENDER_SLOWDOWN_S)
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return False, "ffmpeg binary not found"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(source),
        "-t", f"{duration_s}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dest),
    ]
    last_stderr = ""
    for attempt in range(6):
        if attempt > 0:
            # Backoff 0.1, 0.2, 0.4, 0.8, 1.6 s ≈ 3 s cumulative.
            time.sleep(0.1 * (2 ** (attempt - 1)))
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30.0
            )
        except subprocess.TimeoutExpired as e:
            return False, f"ffmpeg timed out: {e}"
        last_stderr = res.stderr or ""
        if res.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            return True, last_stderr
        # "moov atom not found" = file still flushing. Retry.
        if "moov atom not found" in last_stderr or "Invalid data" in last_stderr:
            continue
        # Any other error is not flush-related; fail fast.
        return False, last_stderr
    return False, last_stderr


def _find_ffmpeg() -> Optional[str]:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


# ── test helpers ──────────────────────────────────────────────────────
def set_render_slowdown(seconds: Optional[float]) -> None:
    """Install an artificial per-render sleep. Used only by R3
    saturation tests — None (the default) disables."""
    global _RENDER_SLOWDOWN_S
    _RENDER_SLOWDOWN_S = seconds
