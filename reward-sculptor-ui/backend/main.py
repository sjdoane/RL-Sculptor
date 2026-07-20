"""FastAPI entry point.

Composition root:
  - Loads Settings (env + .env).
  - Runs the cloud-sync guard; aborts with SystemExit(2) on refusal.
  - Creates the ProjectStore (directory-backed registry).
  - Mounts routes and CORS.
  - Exposes GET /health for sculptor-import status.

Start with:
  uv run uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

from backend.config import Settings, abort_on_cloud_sync
from backend.models.project import HealthResponse
from backend.routes import dashboard as dashboard_routes
from backend.routes import jobs as jobs_routes
from backend.routes import kg as kg_routes
from backend.routes import library as library_routes
from backend.routes import metrics as metrics_routes
from backend.routes import missions as missions_routes
from backend.routes import physics as physics_routes
from backend.routes import policies as policies_routes
from backend.routes import projects as projects_routes
from backend.routes import references as references_routes
from backend.routes import rewards as rewards_routes
from backend.routes import robot as robot_routes
from backend.routes import reports as reports_routes
from backend.routes import runs as runs_routes
from backend.routes import saved as saved_routes
from backend.routes import system as system_routes
from backend.routes import trash as trash_routes
from backend.routes import worlds as worlds_routes
from backend.services.job_manager import JobManager
from backend.services import sculptor_bridge
from backend.services.api_key_store import load_saved_key
from backend.services.project_store import ProjectStore


log = logging.getLogger("reward-sculptor-ui")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the FastAPI application.

    Tests call this with an explicit `settings` that points at a tmp
    projects root. In production, uvicorn imports the module-level
    `app` below, which builds with the default Settings (env-driven).
    """
    s = settings or Settings()
    abort_on_cloud_sync(s)

    root = s.resolved_projects_root
    root.mkdir(parents=True, exist_ok=True)
    # A key entered in Settings becomes available immediately when saved and
    # is restored on later launches.  An explicit process environment value
    # always wins over the UI-saved copy.
    load_saved_key(root)
    store = ProjectStore(root)

    app = FastAPI(
        title="Reward Sculptor UI",
        version="0.1.0",
        description=(
            "Localhost control panel for reward-sculptor. v0.1 exposes "
            "project-lifecycle endpoints only."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = s
    app.state.project_store = store
    app.state.job_manager = JobManager()

    @app.on_event("startup")
    async def _bind_loop() -> None:  # pragma: no cover
        import asyncio as _asyncio
        app.state.job_manager.bind_loop(_asyncio.get_running_loop())

    @app.on_event("startup")
    async def _bootstrap_shared_kg() -> None:
        """Seed the user-wide shared KG from the bundled pre-extracted
        sqlite on first launch. No-op if the shared KG already exists,
        the bundled DB is missing, or `RS_KG_PATH` is set (tests +
        ad-hoc redirects should not auto-seed a tmp DB).

        The bundled DB lives at
        `<RewardSculptor repo>/examples/kg_preextracted.db` and is
        regenerated via `scripts/regenerate_kg.sh` when
        `examples/kg_seeds_global.yml` changes.
        """
        import os as _os
        import shutil

        # Never auto-seed when a caller has redirected the KG path —
        # avoids polluting test tmp dirs or the user's custom path.
        if _os.environ.get("RS_KG_PATH") or _os.environ.get("SCULPTOR_KG_PATH"):
            return

        try:
            from backend.services.kg_store import shared_kg_db_path
            shared = shared_kg_db_path()
            if shared.exists():
                return
            try:
                import sculptor
            except ImportError:
                log.info(
                    "sculptor package unavailable; skipping KG bootstrap"
                )
                return
            sculptor_root = (
                Path(sculptor.__file__).resolve().parent.parent
            )
            bundled = sculptor_root / "examples" / "kg_preextracted.db"
            if not bundled.is_file():
                log.info(
                    "bundled kg_preextracted.db not at %s — shared KG "
                    "starts empty. Run scripts/regenerate_kg.sh to "
                    "materialise it.", bundled,
                )
                return
            shared.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled, shared)
            log.info(
                "bootstrapped shared KG: copied %s -> %s "
                "(%d bytes)",
                bundled, shared, shared.stat().st_size,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "KG bootstrap failed: %s: %s", type(e).__name__, e
            )

    @app.on_event("startup")
    async def _prewarm_embedding_model() -> None:
        """§Ship-10 — the sentence-transformers model (all-MiniLM-L6-v2)
        takes 60-120s to download + init on a cold WSL2 cache, and the
        first `query_semantic` call pays that cost inline. Pre-Ship-10
        this made reward-prompt edits look "stuck on Claude is writing"
        for minutes after every `./run.sh` restart.

        Fire-and-forget on a worker thread so uvicorn startup doesn't
        block. `_get_embedder` (sculptor/kg/query.py) holds a
        `threading.Lock` so a reward-prompt worker hitting the same
        call mid-prewarm waits for prewarm to finish instead of racing
        into a second `SentenceTransformer(...)` init, which on WSL2
        can deadlock on the huggingface_hub cache FileLock (observed
        as a 5-minute hang on the KG preview step, 2026-04-23 03:22).

        The task reference is stashed on `app.state` so Python's weak-
        ref handling of `asyncio.create_task` (3.11+) doesn't GC the
        prewarm mid-flight.
        """
        import asyncio as _asyncio
        import os as _os

        if _os.environ.get("RS_SKIP_EMBEDDER_PREWARM") == "1":
            return

        def _load_embedder() -> None:
            try:
                from sculptor.kg.query import _get_embedder
                _get_embedder()
                log.info("embedding model pre-warmed")
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "embedder pre-warm failed (will load lazily on "
                    "first query): %s: %s", type(e).__name__, e,
                )

        # Pin the task so it isn't GC'd before the executor picks it up.
        app.state.embedder_prewarm_task = _asyncio.create_task(
            _asyncio.to_thread(_load_embedder),
            name="embedder-prewarm",
        )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        ok = sculptor_bridge.sculptor_ok()
        return HealthResponse(
            status="ok" if ok else "degraded",
            sculptor_ok=ok,
            sculptor_error=sculptor_bridge.sculptor_error(),
            projects_root=str(s.resolved_projects_root),
        )

    app.include_router(projects_routes.router)
    app.include_router(robot_routes.router)
    app.include_router(rewards_routes.router)
    app.include_router(kg_routes.router)
    app.include_router(runs_routes.router)
    app.include_router(policies_routes.router)
    app.include_router(reports_routes.router)
    app.include_router(jobs_routes.router)
    app.include_router(jobs_routes.ws_router)
    app.include_router(dashboard_routes.router)
    app.include_router(system_routes.router)
    app.include_router(library_routes.router)
    app.include_router(physics_routes.router)
    app.include_router(missions_routes.router)
    app.include_router(missions_routes.ws_router)
    app.include_router(metrics_routes.router)
    app.include_router(trash_routes.router)
    app.include_router(saved_routes.router)
    app.include_router(references_routes.router)
    app.include_router(worlds_routes.router)
    return app


# Module-level app for uvicorn.
app = create_app()
