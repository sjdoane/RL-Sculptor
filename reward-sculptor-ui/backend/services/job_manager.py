"""In-process job registry backing the async endpoints.

Design goals:
  - No external broker. Jobs live in-memory; backend restart drops
    history. This is fine for a localhost-only UI.
  - Cooperative cancellation via asyncio.Event — jobs that wrap sync
    workloads poll it between chunks of work.
  - Each job's handler is an async callable. For sync sculptor calls
    (ingest, extract, build_kg_html) it wraps via asyncio.to_thread.

Used by: routes/kg.py (seeds ingest+extract, graph render),
routes/rewards.py (not in Prompt 7 — placeholder for future replays),
routes/jobs.py (GET /jobs/{job_id}).
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from backend.models.kg import JobDetail, JobKind, JobStatus, JobSummary


JobCallable = Callable[["Job", "asyncio.Event"], Awaitable[Optional[dict[str, Any]]]]


@dataclass
class Job:
    job_id: str
    kind: JobKind
    project_slug: Optional[str]
    status: JobStatus = "queued"
    progress: Optional[float] = None
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    params: dict[str, Any] = field(default_factory=dict)

    # ── event stream + log ring (used by sculpt_run jobs) ──────────────
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    # In-memory ring for fast replay on WS reconnect; matches the
    # Prompt 8 spec of "last 200 lines in memory". Lines older than this
    # live on disk only (see RunContext.log_file_path).
    log_ring: deque[str] = field(
        default_factory=lambda: deque(maxlen=200), repr=False,
    )

    # ── internal ───────────────────────────────────────────────────────
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _cancel: Optional[asyncio.Event] = field(default=None, repr=False)
    _subscribers: list["asyncio.Queue[dict[str, Any]]"] = field(
        default_factory=list, repr=False,
    )
    _seq_counter: int = field(default=0, repr=False)

    def to_summary(self) -> JobSummary:
        return JobSummary(
            job_id=self.job_id,
            kind=self.kind,
            project_slug=self.project_slug,
            status=self.status,
            progress=self.progress,
            message=self.message,
            started_at=self.started_at,
            ended_at=self.ended_at,
            error=self.error,
        )

    def to_detail(self) -> JobDetail:
        return JobDetail(
            job_id=self.job_id,
            kind=self.kind,
            project_slug=self.project_slug,
            status=self.status,
            progress=self.progress,
            message=self.message,
            started_at=self.started_at,
            ended_at=self.ended_at,
            error=self.error,
            params=dict(self.params),
            result=self.result,
        )

    # ── event stream wiring ────────────────────────────────────────────
    def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append `event` to the job's event log and broadcast to every
        WS subscriber. Adds `seq` + `ts` fields if not present.

        For `log_line` events the `text` also lands in `log_ring` (the
        in-memory reconnect buffer).
        """
        self._seq_counter += 1
        enriched = dict(event)
        enriched.setdefault("seq", self._seq_counter)
        enriched.setdefault("ts", datetime.now(tz=timezone.utc).isoformat())
        self.events.append(enriched)
        if enriched.get("type") == "log_line":
            text = enriched.get("text")
            if isinstance(text, str):
                self.log_ring.append(text)
        # Broadcast. Drop events for back-pressured subscribers rather
        # than stalling the producer — WS clients are expected to keep
        # up; a laggy client loses live tail but can reconnect to
        # replay from `events`.
        for q in self._subscribers:
            try:
                q.put_nowait(enriched)
            except asyncio.QueueFull:
                pass
        return enriched

    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]":
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10_000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[dict[str, Any]]") -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()  # guards _jobs dict
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Stash the main event loop so sync endpoint handlers can
        submit jobs from anyio worker threads."""
        self._bound_loop = loop

    # ── submit ─────────────────────────────────────────────────────────
    def submit(
        self,
        kind: JobKind,
        project_slug: Optional[str],
        fn: JobCallable,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> Job:
        job_id = f"job_{secrets.token_hex(8)}"
        cancel = asyncio.Event()
        job = Job(
            job_id=job_id,
            kind=kind,
            project_slug=project_slug,
            params=dict(params or {}),
        )
        job._cancel = cancel
        with self._lock:
            self._jobs[job_id] = job

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._bound_loop
            if loop is None:
                raise RuntimeError(
                    "JobManager.submit called with no running loop; "
                    "either await in an async handler or call "
                    "JobManager.bind_loop(loop) at app startup."
                ) from None

        async def _runner() -> None:
            job.status = "running"
            job.started_at = datetime.now(tz=timezone.utc)
            try:
                result = await fn(job, cancel)
            except asyncio.CancelledError:
                job.status = "stopped"
                job.error = "cancelled"
                raise
            except Exception as e:  # noqa: BLE001
                job.status = "errored"
                job.error = f"{type(e).__name__}: {e}"
            else:
                # If the fn set status itself (e.g., partial failure),
                # respect that — otherwise mark completed.
                if job.status == "running":
                    job.status = "completed"
                if isinstance(result, dict):
                    job.result = result
            finally:
                job.ended_at = datetime.now(tz=timezone.utc)

        if asyncio.get_event_loop_policy()._local._loop is loop and loop.is_running():  # type: ignore[attr-defined]
            # Called from within the loop's own thread (async handler).
            job._task = loop.create_task(_runner())
        else:
            # Worker-thread fallback: schedule onto the bound loop.
            # `run_coroutine_threadsafe` returns a concurrent.futures.Future,
            # which supports .cancel() but is not an asyncio.Task. That's
            # fine — we never await this handle in-thread.
            job._task = asyncio.run_coroutine_threadsafe(_runner(), loop)  # type: ignore[assignment]
        return job

    # ── queries ────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(
        self,
        *,
        kind: Optional[JobKind] = None,
        project_slug: Optional[str] = None,
        status: Optional[JobStatus] = None,
    ) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind is not None:
            jobs = [j for j in jobs if j.kind == kind]
        if project_slug is not None:
            jobs = [j for j in jobs if j.project_slug == project_slug]
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.started_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return jobs

    def has_active_sculpt_run(self, slug: str) -> bool:
        """True if the project has a sculpt_run job in running / queued
        state. Used by the project-status computation in project_store."""
        for j in self.list(kind="sculpt_run", project_slug=slug):
            if j.status in ("running", "queued"):
                return True
        return False

    def has_any_active_sculpt_run(self) -> bool:
        """True if ANY project has a sculpt_run in running / queued state.
        Used by the preview / reports routes to distinguish GPU-busy 503s
        from genuine 500s when a render fails during a concurrent run
        (§7.6). Unlike `has_active_sculpt_run(slug)` this is cross-project
        because the GPU conflict is process-wide, not per-project."""
        for j in self.list(kind="sculpt_run"):
            if j.status in ("running", "queued"):
                return True
        return False

    # ── control ────────────────────────────────────────────────────────
    def stop(self, job_id: str) -> Optional[Job]:
        job = self.get(job_id)
        if job is None or job._cancel is None:
            return None
        # Flip the cooperative cancel flag from the loop thread so any
        # code awaiting the Event sees it atomically.
        if self._bound_loop is not None and self._bound_loop.is_running():
            self._bound_loop.call_soon_threadsafe(job._cancel.set)
        else:
            try:
                job._cancel.set()
            except RuntimeError:
                pass
        task = job._task
        if task is not None:
            try:
                done_fn = getattr(task, "done", lambda: True)
                if not done_fn():
                    task.cancel()
            except Exception:  # noqa: BLE001
                pass
        return job

    async def wait(self, job_id: str, timeout: float = 60.0) -> Optional[Job]:
        """Test helper: block until the job terminates. Returns the Job
        or None if it never existed."""
        job = self.get(job_id)
        if job is None or job._task is None:
            return job
        deadline = time.time() + timeout
        while job.status in ("queued", "running"):
            if time.time() > deadline:
                break
            await asyncio.sleep(0.05)
        return job
