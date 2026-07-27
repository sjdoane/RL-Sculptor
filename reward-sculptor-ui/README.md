# Reward Sculptor UI

A localhost control panel for [Reward Sculptor](../RewardSculptor).
FastAPI backend + React + Vite frontend. Create projects, pick or
upload a robot, browse the knowledge graph, launch sculpt runs, and
watch them live — all in one tab.

![Dashboard](docs/screenshots/dashboard.png)

> Screenshots are committed manually by the author — the file above is
> a zero-byte placeholder on a fresh clone. See
> [`docs/screenshots/README.md`](docs/screenshots/README.md) for the
> full list. Replace `dashboard.png` with a real capture when you ship.

## One-command install

```bash
cd reward-sculptor-ui
uv sync
pnpm install --dir frontend
```

On Windows, both commands may want `UV_LINK_MODE=copy` if your repo is
inside OneDrive — see [Windows + OneDrive notes](#windows--onedrive-notes).

## One-command run

```bash
# POSIX (macOS / Linux / WSL)
./run.sh

# Windows PowerShell
pwsh ./run.ps1
```

The script:

- checks `uv` + `pnpm` are on PATH, `.venv` + `frontend/node_modules`
  exist, and port 8000 is free (with clear messages if any check fails);
- reports when no shell/repository API key is present; a key saved from
  **Settings → Anthropic API** is loaded automatically and never returned
  by the API;
- starts the FastAPI backend on `127.0.0.1:$RS_PORT` (default 8000),
  waits for `/health` to 200;
- starts the Vite dev server on `127.0.0.1:5173`;
- opens `http://127.0.0.1:5173` in your default browser;
- traps Ctrl+C and kills both subprocesses cleanly.

## Manual run (without the script)

```bash
# terminal 1
uv run uvicorn backend.main:app --reload --port 8000
# terminal 2
pnpm --dir frontend dev
# then open http://localhost:5173
```

## Tests

```bash
uv run pytest backend/tests/ -v
pnpm --dir frontend run build  # typecheck + production bundle
```

The backend imports `sculptor` as a library (path-install of
`../RewardSculptor`). It never writes to that directory — all project
state lands under `$RS_PROJECTS_ROOT`.

## Configuration

Environment variables (all optional):

| Var                        | Default                                                           |
| -------------------------- | ----------------------------------------------------------------- |
| `RS_PROJECTS_ROOT`         | `%LOCALAPPDATA%\reward-sculptor\projects\` (Windows) / `~/.local/share/reward-sculptor/projects/` (POSIX) |
| `RS_ALLOW_CLOUD_SYNC`      | `false` — set to `true` to override the cloud-sync guard          |
| `RS_HOST`                  | `127.0.0.1`                                                       |
| `RS_PORT`                  | `8000`                                                            |
| `RS_CORS_ORIGINS`          | `http://localhost:5173,http://127.0.0.1:5173`                     |

The backend refuses to start if `$RS_PROJECTS_ROOT` resolves to a
directory inside OneDrive / Dropbox / Google Drive / iCloud / Box /
pCloud. sqlite + ffmpeg inside cloud-synced folders is a documented
footgun. Override with `RS_ALLOW_CLOUD_SYNC=true`.

## Features

- **Dashboard** (`/`) — running jobs, recent completed runs with metric
  sparklines, recent KG additions, one-click launch. Empty on a fresh
  install; `Get started` card walks you into project creation.
- **Projects** — create, configure a robot (Gymnasium library pick or
  upload URDF / MJCF / mesh zip), browse rewards + KG + runs + reports
  per-project.
- **Live runs** — launch `sculpt run` as a subprocess, stream stdout +
  filesystem events over WebSocket, virtualized log view, per-iter
  timeline, live primary-metric chart, 2-second rollout clips pushed as
  each iteration completes.
- **Reward editor** — Monaco-backed view of every `v<n>.py`, clone the
  latest into an editable draft, save as a new `human`-authored version
  with server-side AST validation + KG-referenced arxiv IDs.
- **Knowledge graph** — paste arxiv IDs, run ingest + extract as a
  background job, browse papers + extracted techniques + failure
  modes, open the pyvis interactive graph in a modal.
- **Settings** (`/settings`) — save or replace the local Anthropic API key,
  inspect GPU/runtime readiness and paths, toggle theme, and reset UI state.

For a start-to-finish, UI-operated showcase, see the
[lab-call demo runbook](docs/LAB_CALL_DEMO_RUNBOOK.md).

## Endpoints (brief)

- `GET /health` — sculptor import status + configured paths.
- `GET /api/dashboard` — aggregate dashboard data (active jobs, recent
  runs, recent KG additions) in one call.
- `GET /api/system/info` — GPU + API-key + paths info for the Settings
  page.
- `/projects`, `/projects/{slug}/rewards`, `/projects/{slug}/kg/*`,
  `/projects/{slug}/runs`, `/projects/{slug}/reports/*` — per the API
  contract in the design-doc trail.
- `WS /ws/projects/{slug}/runs/{run_id}/events` — live run events.
- `WS /ws/projects/{slug}/runs/{run_id}/frames` — live clip push.

See `backend/tests/test_*.py` for usage patterns.

## Windows + OneDrive notes

This is the durable home for Windows-specific gotchas. Append as new
ones are discovered.

### (a) Cloud-sync guard

The backend refuses to start if `$RS_PROJECTS_ROOT` resolves to a
directory inside a cloud-synced folder (OneDrive, Dropbox, Google
Drive, iCloud Drive, Box, pCloud). The rationale:

- **sqlite**: the KG store opens `kg/graph.db` with WAL/shm sidecar
  files. Cloud-sync agents lock and re-upload these mid-write,
  corrupting the DB in ways sqlite can't recover from.
- **ffmpeg**: the adapter renders rollouts by dumping PNGs to a
  tempdir and running ffmpeg subprocess. Cloud-sync on the project
  dir can hold the output mp4 handle open and corrupt the encode.
- **training checkpoints**: SB3 writes `checkpoint.zip` atomically,
  then cloud-sync opens it for upload. A subsequent overwrite during
  the next iteration collides with the sync lock.

**Override** (at your own risk) with either:

```bash
# env var
RS_ALLOW_CLOUD_SYNC=true uv run uvicorn backend.main:app

# or CLI flag (backend wraps this as the env var)
uv run uvicorn backend.main:app --i-know-what-im-doing  # future CLI
```

Downgrades the fatal abort to a WARN-level log. Don't blame us when
the sqlite DB disintegrates.

### (b) `UV_LINK_MODE=copy`

If either the UI project **or** its `../RewardSculptor` path-install
target lives on OneDrive, `uv sync` will periodically fail with:

```
error: Failed to uninstall: reward_sculptor-0.1.0.dist-info
Caused by: Access is denied (os error 5)
```

OneDrive holds the `.dist-info` directory handle open while syncing.
Workaround: prefix every uv command with `UV_LINK_MODE=copy`:

```bash
UV_LINK_MODE=copy uv sync
UV_LINK_MODE=copy uv run pytest backend/tests/ -v
```

Or set it globally in your shell profile if you work on OneDrive-
resident projects often. The mode tells uv to copy package files
instead of hardlinking — slower install, robust against the lock.

### (c) Host project data outside OneDrive entirely

The backend's platform default (`%LOCALAPPDATA%\reward-sculptor\projects\`)
already puts project *data* outside OneDrive — `%LOCALAPPDATA%` is
local-only by design. Keep it that way. Specifically:

- Don't point `RS_PROJECTS_ROOT` at a OneDrive path even though the
  guard will catch it for you.
- If you want to share project configs across machines, sync the
  *source* (`config.toml`, `kg_seeds.yml`) via Git, not the whole
  `runs/` and `kg/graph.db` state. `runs/` is regeneratable and
  `graph.db` is write-heavy.
- The UI's *code* (this repo) lives on OneDrive and that's fine —
  source files are small, infrequent writes, and `uv sync` with
  `UV_LINK_MODE=copy` handles the `.venv/` rebuild cleanly.

## Future work

Features intentionally deferred from v1 — patches welcome. These are
not "coming soon" promises; each one involves scope decisions the
project hasn't made yet.

- **STEP / STL CAD import** — the robot-upload flow accepts URDF /
  MJCF / mesh zip today. Direct import of mechanical CAD would require
  a separate import pipeline (conversion to MuJoCo mesh + inertial
  parameter inference) that sits outside the UI's current scope.
- **Isaac Gym / Brax adapters in the UI** — the sculptor package
  supports any adapter conforming to `SculptorAdapter` (see
  `../RewardSculptor/docs/adapters.md`). The UI hard-codes the
  `gym_sb3` adapter in its library picker; adding Isaac / Brax
  adapters means exposing per-adapter config templates and pre-flight
  GPU checks.
- **Multi-user auth** — the backend binds to `127.0.0.1` only, there
  is no session model, and `/projects/{slug}` has no owner field.
  Multi-user would need at minimum a login flow, per-user project
  visibility, and a token-gated WS upgrade.
- **Distributed training** — sculpt runs execute as a single
  subprocess per project. Running iterations across multiple
  machines / GPUs would require the run manager to track remote
  workers + a shared-filesystem assumption the backend doesn't make
  today.
- **Autonomous KG research agent** — the KG ingest flow is manual
  (paste arxiv IDs). An agent that reads the project's current
  behavior goal + failure modes and proposes new papers to ingest is
  on the post-v1 list.
