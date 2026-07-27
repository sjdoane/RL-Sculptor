"""Background runner for per-mode reward authoring.

The user picks one mode of a composed reference on the Reference tab and asks
for its reward; Claude writes that mode's two function bodies and nothing else.
Mirrors `reward_jobs.py`'s `asyncio.to_thread` shape because the underlying
sculptor call is synchronous.

Everything about HOW a mode is authored — the stub twin, the graft, the
re-probes, the info-key gate — lives in `sculptor.mode_rewards.author_mode`,
which the `sculpt modes author` CLI calls too. This module is only the job
wrapper: resolve the graph, resolve the file, call it, write the result.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from backend.services.job_manager import Job

# Same budget as a reward prompt-edit, for the same reason: `apply_prompt_edit`
# makes up to two attempts of 180-240s each plus SDK backoff. Authoring uses a
# LARGER token ceiling than a normal edit (`AUTHOR_MAX_TOKENS`), so if anything
# a single attempt runs longer, not shorter.
DEFAULT_MODE_AUTHOR_TIMEOUT_S = float(
    os.environ.get("RS_MODE_AUTHOR_TIMEOUT_S", "900")
)


def run_mode_author_job(
    *,
    project_dir: Path,
    reward_path: Path,
    out_path: Path,
    clip_id: str,
    robot: str,
    mode: str,
    behavior_goal: str = "",
    mode_goal: str = "",
    timeout_s: float | None = None,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Return a job callable that authors one mode into `reward_path` and
    writes the result to `out_path`.

    `out_path` is resolved by the route (not here) so the conflict check and
    the job agree on which file is about to be written.
    """
    budget_s = timeout_s if timeout_s is not None else DEFAULT_MODE_AUTHOR_TIMEOUT_S

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        job.progress = 0.05
        job.message = f"authoring mode {mode!r}"
        job.emit({"type": "log_line",
                  "text": f"[mode_author] start — mode {mode!r} of {clip_id}"})

        loop = asyncio.get_running_loop()

        def _emit_from_worker(ev: dict[str, Any]) -> None:
            # sculptor's `on_event` fires from the to_thread worker; marshal
            # back to the job's loop so `emit` touches its asyncio.Queue on
            # the right thread.
            try:
                loop.call_soon_threadsafe(job.emit, ev)
            except RuntimeError:
                job.emit(ev)

        def _do_author() -> dict[str, Any]:
            _emit_from_worker({
                "type": "log_line",
                "text": "[mode_author] loading adapter + reward_contract"})
            from sculptor.adapters.base import load_adapter
            from sculptor.kg.store import SculptorKG
            from sculptor.mode_rewards import author_mode
            from sculptor.modes import modes_from_composition
            from sculptor.reference import load_clip
            from sculptor.refs.library import clip_dir

            contract = load_adapter(project_dir / "config.toml").reward_contract()
            clip = load_clip(clip_dir(robot, clip_id) / "clip.npz")
            graph = modes_from_composition(clip, clip_id=clip_id)

            from backend.services import kg_store as kg_store_svc
            db_path = kg_store_svc.project_kg_db_path(project_dir)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _emit_from_worker({
                "type": "log_line",
                "text": (f"[mode_author] {len(graph.modes)} modes; delegating "
                         f"to Claude (KG grounding + validate + re-probe)")})

            with SculptorKG(db_path) as store:
                result = author_mode(
                    source=reward_path.read_text(encoding="utf-8"),
                    graph=graph, mode=mode, contract=contract, clip=clip,
                    clip_id=clip_id, behavior_goal=behavior_goal,
                    mode_goal=mode_goal, kg_store=store,
                    on_event=_emit_from_worker)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result["source"], encoding="utf-8")
            authored = result["authored"]
            pending = result["pending"]
            return {
                "mode": mode,
                "clip_id": clip_id,
                "path": str(out_path),
                "filename": out_path.name,
                "prompt": result["prompt"],
                "modes": [{"name": n, "authored": bool(ok)}
                          for n, ok in authored.items()],
                "pending": pending,
                "authored_count": len(authored) - len(pending),
                "mode_count": len(authored),
            }

        job.progress = 0.2
        out = await asyncio.wait_for(asyncio.to_thread(_do_author), budget_s)
        job.progress = 1.0
        done, total = out["authored_count"], out["mode_count"]
        job.message = f"{done}/{total} modes authored"
        job.emit({"type": "log_line",
                  "text": f"[mode_author] {mode}: authored -> {out['filename']} "
                          f"({done}/{total} modes)"})
        return out

    return _runner
