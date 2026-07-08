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
    # §Ship 21: parent job linkage for mission stage runs. When set,
    # this job represents one stage of a mission_execute run; its
    # events mirror what the parent's stdout streamer detected for
    # the matching stage_name. None for top-level sculpt_run /
    # mission_decompose / mission_execute jobs.
    parent_id: Optional[str] = None

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
            parent_id=self.parent_id,
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
            parent_id=self.parent_id,
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
        parent_id: Optional[str] = None,
    ) -> Job:
        job_id = f"job_{secrets.token_hex(8)}"
        cancel = asyncio.Event()
        job = Job(
            job_id=job_id,
            kind=kind,
            project_slug=project_slug,
            params=dict(params or {}),
            parent_id=parent_id,
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

    # §Ship 21: register a job that has no runner of its own. Used by
    # `mission_jobs._stream_stdout` when a stage_started event fires
    # inside a parent mission_execute subprocess — the work is already
    # happening (it's a fragment of the parent's lifetime), but we
    # want the stage to appear as a first-class entry in /runs. The
    # caller (mission_jobs) drives lifecycle via emit() + by setting
    # status / started_at / ended_at directly.
    def register_passive_job(
        self,
        kind: JobKind,
        project_slug: Optional[str],
        *,
        params: Optional[dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> Job:
        job_id = f"job_{secrets.token_hex(8)}"
        job = Job(
            job_id=job_id,
            kind=kind,
            project_slug=project_slug,
            params=dict(params or {}),
            parent_id=parent_id,
            status="running",
            started_at=datetime.now(tz=timezone.utc),
        )
        # Passive jobs have no _task; they cannot be cancelled
        # individually (cancelling the parent kills the subprocess
        # which terminates all stages). _cancel stays None.
        with self._lock:
            self._jobs[job_id] = job
        return job

    def has_active_sculpt_run(self, slug: str) -> bool:
        """True if the project has a sculpt_run job in running / queued
        state. Used by the project-status computation in project_store."""
        for j in self.list(kind="sculpt_run", project_slug=slug):
            if j.status in ("running", "queued"):
                return True
        return False

    def active_jobs_for_project(self, slug: str) -> list[Job]:
        """Every job of ANY kind for this project in running / queued
        state (chunk A1) — used by `DELETE /projects/{slug}` to refuse
        moving a project into the trash out from under a live job
        (sculpt_run, mission_decompose, mission_execute, kg jobs, …).
        Unlike `has_active_sculpt_run` this isn't scoped to one kind,
        since any in-flight job holding file handles / writing under
        the project dir makes a concurrent move unsafe."""
        return [
            j for j in self.list(project_slug=slug)
            if j.status in ("running", "queued")
        ]

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

    def has_any_active_gpu_job(self) -> bool:
        """§Ship 18a — generalization of `has_any_active_sculpt_run` to
        include `mission_execute` jobs. A mission_execute spawns its
        own per-stage sculpt_run subprocesses, so it competes for the
        SAME GPU as a top-level sculpt_run. Routes that need GPU-busy
        guards (POST /projects/.../runs, POST /projects/.../missions/
        */run) call this rather than the sculpt-only variant."""
        for j in self.list():
            if j.kind in ("sculpt_run", "mission_execute") and j.status in (
                "running", "queued",
            ):
                return True
        return False

    def active_mission_job(
        self, project_slug: str, mission_slug: str,
    ) -> Optional[Job]:
        """§Ship 18a — the in-flight `mission_decompose` or
        `mission_execute` job for a specific (project, mission) pair,
        or None. Used by the routes layer to (a) refuse concurrent
        decompose/execute on the same mission, (b) attach
        `active_job_id` to MissionSummary / MissionDetail responses,
        (c) pick the job whose events the mission WS route streams.

        Deliberately narrow to these two kinds — callers (b) and (c)
        need "the one job that represents the mission's own headline
        lifecycle," not every job that happens to touch it. Use
        `active_mission_scoped_job` for a broader "is ANY mission-
        scoped job live" concurrency guard.
        """
        for j in self.list(project_slug=project_slug):
            if j.kind not in ("mission_decompose", "mission_execute"):
                continue
            if j.params.get("mission_slug") != mission_slug:
                continue
            if j.status in ("running", "queued"):
                return j
        return None

    # Kinds whose params carry a `mission_slug` scoping them to one
    # mission. Kept as one list so `active_mission_scoped_job` and any
    # future caller stay in sync as new mission-scoped job kinds are
    # added (e.g. a future per-stage job kind).
    _MISSION_SCOPED_KINDS = (
        "mission_decompose",
        "mission_execute",
        "mission_stage_run",
        "mission_stage_metric_regen",
    )

    def active_mission_scoped_job(
        self, project_slug: str, mission_slug: str,
    ) -> Optional[Job]:
        """The in-flight job of ANY mission-scoped kind — decompose,
        execute, a per-stage run, or a metric regenerate — for a
        specific (project, mission) pair, or None.

        Broader than `active_mission_job`: used where the concurrency
        hazard is "some write touching this mission's mission.json (or
        a stage under it) is in flight," not just "the mission's own
        headline decompose/execute lifecycle." `POST .../run` uses this
        directly so it also refuses to start `mission_execute` while a
        `mission_stage_metric_regen` for the SAME mission is mid-save
        (both call `save_mission`, non-atomic last-write-wins).
        """
        for j in self.list(project_slug=project_slug):
            if j.kind not in self._MISSION_SCOPED_KINDS:
                continue
            if j.params.get("mission_slug") != mission_slug:
                continue
            if j.status in ("running", "queued"):
                return j
        return None

    def active_mission_metric_regen_job(
        self, project_slug: str, mission_slug: str,
    ) -> Optional[Job]:
        """The in-flight `mission_stage_metric_regen` job for a
        specific (project, mission) pair (any stage), or None.

        Split out from `active_mission_scoped_job` because
        `POST .../metric/regenerate` needs to distinguish "another
        regen for this mission is running" (its own dedicated 409
        message, naming the other stage) from "a mission_stage_run for
        some OTHER stage happens to be live" (not a conflict for this
        route — only a stage_run for THIS SAME stage is, handled
        separately by that route's existing per-stage loop).
        """
        for j in self.list(project_slug=project_slug):
            if j.kind != "mission_stage_metric_regen":
                continue
            if j.params.get("mission_slug") != mission_slug:
                continue
            if j.status in ("running", "queued"):
                return j
        return None

    # ── control ────────────────────────────────────────────────────────
    def stop(self, job_id: str) -> Optional[Job]:
        job = self.get(job_id)
        if job is None:
            return None
        # §Ship 21: passive jobs (mission_stage_run) have no _cancel/
        # _task because they're driven by the parent's subprocess. To
        # honor a Stop click on a stage row, route the cancellation
        # to the parent mission_execute job whose subprocess kill
        # terminates all child stages. Audit-fix CRITICAL #2: prior
        # to this, stop() returned None on a passive job so the user
        # got a "Kill signal sent" toast but the run kept training.
        if job._cancel is None and job.parent_id is not None:
            parent = self.get(job.parent_id)
            if parent is not None and parent._cancel is not None:
                return self.stop(parent.job_id)
            return None
        if job._cancel is None:
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
