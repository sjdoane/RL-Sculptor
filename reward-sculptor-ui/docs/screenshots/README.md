# Screenshots

`library-idle.jpg` is an unedited browser capture of `/library`, taken on
2026-10-28 from public source revision
`52a149a8172e2b84345d1b7f4e8b352fefdc46d6` on macOS, Python 3.12.14,
Node 24.20.0 and pnpm 9.12.0. The viewport is 1124 × 720.

This is an idle UI inspection, using a new empty project directory, no API
key and no installed torch, simulator or embedding model. The library returns
63 catalogue entries; the default filters show three Gymnasium entries.
The page reports zero ready to train and no GPU. Its robot pictures are
bundled catalogue thumbnails, not rollouts from this session. No project,
training, ingestion, preview rendering or provider operation was started.

## Reproduce the idle view

Use a fresh public clone without `.env` files in the clone or any ancestor
directory: the core package searches those locations when imported. These
POSIX commands create a separate UI inspection environment. They do **not**
verify the full `uv sync` installation or make training available.

From `reward-sculptor-ui/`, install only the packages used by this capture,
with every version selected from the existing backend lockfile:

```bash
uv venv --python 3.12 .venv
.venv/bin/python - <<'PY'
import subprocess
import tomllib

names = set('''annotated-doc annotated-types anyio click fastapi filelock
h11 idna numpy nvidia-ml-py packaging pydantic pydantic-core
pydantic-settings python-dotenv python-multipart pyyaml starlette
typing-extensions typing-inspection uvicorn watchfiles'''.split())
with open('uv.lock', 'rb') as handle:
    packages = tomllib.load(handle)['package']
specs = [f"{p['name']}=={p['version']}" for p in packages if p['name'] in names]
subprocess.run(['uv', 'pip', 'install', '--python', '.venv/bin/python',
                *specs], check=True)
PY
(cd frontend && CI=true corepack pnpm install --frozen-lockfile)
```

Start the backend in terminal 1, still from `reward-sculptor-ui/`:

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import tempfile

ui = Path.cwd().resolve()
core = ui.parent / 'RewardSculptor'
assert not any((p / '.env').exists() for p in (ui, core, *core.parents)), \
    'Use a clean clone without ancestor .env files'
state = Path(tempfile.mkdtemp(prefix='rl-sculptor-ui-capture-'))
print(f'Isolated capture state: {state}', flush=True)
env = {
    'PATH': os.environ['PATH'],
    'PYTHONPATH': str(core),
    'RS_PROJECTS_ROOT': str(state / 'projects'),
    'RS_KG_PATH': str(state / 'kg' / 'graph.db'),
    'RS_TRASH_ROOT': str(state / 'trash'),
    'RS_SKIP_EMBEDDER_PREWARM': '1',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
}
python = str(ui / '.venv' / 'bin' / 'python')
os.execve(python, [python, '-m', 'uvicorn', 'backend.main:app',
                   '--host', '127.0.0.1', '--port', '8765'], env)
PY
```

The explicit environment excludes inherited credentials. The new projects
root also isolates saved API keys under `_settings`; the separate KG path
prevents reading or seeding the normal shared knowledge graph. The prewarm
flag skips the embedding-model download. Nothing overrides `HOME`.

In terminal 2, from `reward-sculptor-ui/`:

```bash
cd frontend
env -i PATH="$PATH" RS_BACKEND_URL=http://127.0.0.1:8765 \
  node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://127.0.0.1:5173/library` in a fresh browser context. Leave the
default filters selected and wait for the catalogue to load. The sidebar
automatically makes read-only system-availability requests. Do not select a
robot, create a project or start a job for this capture. Capture the actual
page without editing its text or injecting API responses. Close the tab and
press Ctrl+C in both terminals when finished.

The capture verified real HTTP 200 responses from `/health`, `/dashboard`,
`/library/robots`, `/system/gpu`, `/system/info` and the three visible thumbnail
routes. `/health` reporting `sculptor_ok` proves import availability only.
The frontend typecheck and production build passed; full backend tests,
simulator execution, training and model/provider behavior were not verified.

## Future captures

Suggested captures:

| File                  | Contents                                           |
| --------------------- | -------------------------------------------------- |
| `dashboard.png`       | `/` route — dashboard with active jobs + recent runs |
| `project-overview.png`| `/projects/<slug>` Overview tab — RobotViewer + metadata |
| `runs-tab.png`        | `/projects/<slug>` Runs tab — live log + timeline + metric chart |
| `kg-tab.png`          | `/projects/<slug>` Knowledge Graph tab — paper browser + graph modal |
| `reports-tab.png`     | `/projects/<slug>` Reports tab — final report + timelapse |

The existing PNG files are zero-byte placeholders. Do not embed them or
populate the suggested result views with invented data. Replace a placeholder
only with a real capture whose source, state and limitations are recorded.
