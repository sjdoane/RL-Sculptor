"""Background runner for the `mission_save` job kind (chunk A4).

Missions already auto-archive incrementally as they run (A3's
orchestrator hooks). This runner backs the MANUAL "save a copy now"
action exposed by `POST /projects/{slug}/missions/{mission_slug}/save`
— it always mints a fresh timestamped entry (`incremental=False`) so a
user-triggered save never silently merges into whatever entry the
mission's own auto-archive is currently writing.

Single-call shape (unlike mission_jobs.py's decompose/execute split):
`archive_mission` is synchronous filesystem copying, no subprocess, no
Claude call — `asyncio.to_thread` is enough to keep the event loop
unblocked while it walks the mission tree.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from backend.services.job_manager import Job


def run_mission_save_job(
    *,
    project_dir: Path,
    project_slug: str,
    mission_slug: str,
    pinned: Optional[dict[str, set[int]]] = None,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Async runner that archives `mission_slug` into the durable saved
    store via `sculptor.archive.archive_mission`.

    Fail-loud: any exception (missing mission dir, bad mission.json,
    IO error) propagates out of the runner so `JobManager.submit`'s
    wrapper marks the job `errored` with the exception text — archiving
    a mission the caller explicitly asked to save must never look like
    a silent no-op.
    """

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        job.progress = 0.05
        job.message = f"saving mission {mission_slug}"
        job.emit({
            "type": "mission_save_started",
            "mission_slug": mission_slug,
            "project_slug": project_slug,
        })

        def _do_archive() -> "Any":
            from sculptor.archive import archive_mission, saved_root
            from backend.services import mission_store

            mission_dir = mission_store.mission_dir(project_dir, mission_slug)
            if not (mission_dir / "mission.json").is_file():
                raise FileNotFoundError(
                    f"{mission_dir} has no mission.json — cannot save "
                    f"mission {mission_slug!r}"
                )
            return archive_mission(
                mission_dir,
                saved_root(),
                project_slug=project_slug,
                pinned=pinned,
                # Manual save action: always a NEW timestamped entry,
                # distinct from whatever entry the mission's own
                # incremental auto-archive is (or was) writing into.
                incremental=False,
            )

        result = await asyncio.to_thread(_do_archive)

        job.progress = 1.0
        job.message = f"saved mission {mission_slug}"
        payload = {
            "mission_slug": mission_slug,
            "project_slug": project_slug,
            "entry_id": result.entry_dir.name,
            "entry_dir": str(result.entry_dir),
            "total_bytes": result.total_bytes,
            "dropped_bytes": result.dropped_bytes,
            "kept_checkpoints": result.kept_checkpoints,
        }
        job.emit({"type": "mission_saved", **payload})
        return payload

    return _runner
