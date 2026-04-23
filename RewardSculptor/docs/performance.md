# Performance baseline (M6)

All measurements on RTX 5070 Laptop GPU (8 GiB VRAM, sm_120), WSL2
Ubuntu 24.04, Ryzen-class CPU, 32 GiB RAM, Python 3.13.5, mjlab 1.3.0,
torch 2.11.0+cu130. Taken 2026-04-20.

## Endpoint latency (FastAPI, in-process via TestClient)

| Endpoint | Cold | Warm (cache hit) | Target | Notes |
|----------|------|------------------|--------|-------|
| `GET /health` | 6 ms | 6 ms | < 100 ms | ✓ |
| `GET /library/categories` | 3 ms | 3 ms | < 100 ms | ✓ static enum |
| `GET /library/adapters` | 7 ms | 7 ms | < 100 ms | ✓ |
| `GET /library/robots` (full list) | **10.8 s** | 3-4 ms | < 500 ms | ⚠ cold path pays the D-guard (see below) |
| `GET /library/robots?category=...` (after warm) | 5 ms | 5 ms | < 100 ms | ✓ |
| `GET /system/gpu` | **3.5 s** | 6 ms | < 500 ms | ⚠ cold path pays pynvml + torch.cuda init |
| `GET /projects` (empty) | 9 ms | — | < 500 ms | ✓ |

### The 10.8 s library-first-request

The library loader's D-guard (cross-reference `mjlab_ready` task IDs
against `mjlab.tasks.registry.list_tasks()` via subprocess) runs
lazily on the first call to `list_robots()` that needs it. Per design
(MJLAB_PIVOT_DESIGN §7 lazy-import rule) this work doesn't happen at
FastAPI startup — which keeps cold-start < 1 s — and the cost is paid
once per backend process.

**Impact on UX:** the first user action that hits `/library/robots`
(usually landing on the Library tab within a few seconds of starting
the UI) blocks for ~10 s. Acceptable for local-dev; if a user waits
10 min before clicking the tab, the delay is surprising.

**Mitigation options (not implemented; post-M6 if it becomes painful):**
- Pre-warm at uvicorn startup via a FastAPI `@app.on_event("startup")`
  hook that fires `get_library().list_robots()` in a background task.
  Moves the 10 s hit to backend boot (which the user already sees).
- Cache the task list on disk at `~/.cache/sculptor/mjlab_tasks.json`
  keyed by `mjlab.__version__`. Startup reads the cache in microseconds;
  subprocess only fires on version change.

### The 3.5 s GPU-first-request

First `gpu_monitor.get_live_snapshot()` call pays `pynvml.nvmlInit()`
(~2.5 s on this host) + first `torch.cuda.get_device_properties()`
(~1 s to initialise CUDA context). 2 s cache TTL means any subsequent
poll within ~2 s is free. The Settings page's 5 s refresh interval
keeps live data flowing at low cost.

## Backend cold-start

| What | Time |
|------|------|
| `create_app()` 3x mean | 0.56 s |
| uvicorn full boot (including `@app.on_event("startup")`) | ~1 s |

Target: < 3 s. ✓ (lazy-import discipline holding.)

## mjlab training throughput baseline (Scenario B)

Measured via `pytest -m gpu` — fixture `mjlab_go1_checkpoint`:

| Parameter | Value |
|-----------|-------|
| Task | `Mjlab-Velocity-Flat-Unitree-Go1` |
| num_envs | 1024 |
| max_iterations | 100 |
| num_steps_per_env (rsl_rl default) | 24 |
| Wall-clock | 89 s |
| Env steps / sec | **~27,500** |
| Iters / sec | 1.1 |
| Peak VRAM | ~3.2 GiB (per `nvidia-smi` during run) |
| Throughput per SM | ~27,500 / 36 = ~760 env-steps/sec/SM |

Subsequent fixture loads: `torch.load(go1_smoke_checkpoint.pt)` in
3.25 s wall-clock (including pytest collection + fixture cache check).

### Reference numbers (from mjlab paper / playground README)

- mjlab_playground Go1 getup: ~2 min on RTX 5090 for convergence.
- Our RTX 5070 Laptop at 1024 envs: ~100 iters/min; 3000 iters to
  convergence extrapolates to ~30 min. Consistent with a 5090 being
  ~4x faster for this workload.

## Backend test suite

| Suite | Tests | Wall-clock |
|-------|-------|------------|
| `reward-sculptor-ui/backend/tests/` | 124 | 35 s |
| `RewardSculptor/tests/` (no GPU) | 76 + 1 skipped | 13 s |
| `RewardSculptor/tests/test_mjlab_gpu.py` (cached fixture) | 2 | 3.25 s |

Total: **203 tests** across the two projects (including the 2 GPU
tests). CI cost is 13 s + 35 s = 48 s for the non-GPU suite.

## Memory footprint

### Backend (uvicorn, idle)

| Process | Resident |
|---------|----------|
| uvicorn main worker | 280 MB |
| uvicorn + 1 active job | 340 MB |

Post-D-guard (mjlab import subprocess fires once) the backend RSS
stays flat — the subprocess exits so its CUDA init cost doesn't
persist.

### Library thumbnails on disk

63 PNGs × average 45 KiB = **2.8 MiB** committed to
`frontend/public/robots/`.

### robot_descriptions cache

Menagerie + Unitree ROS clones cached at
`~/.local/share/robot_descriptions/` = **~700 MiB** (one-time).

## Missed targets + next steps

| Target | Actual | Plan |
|--------|--------|------|
| `/library/robots` < 500 ms cold | 10.8 s | Pre-warm at backend startup OR cache mjlab task list on disk. Track as post-M6. |
| `/system/gpu` < 500 ms cold | 3.5 s | Accept the once-per-process cost; cache hides it after. |

Every other target is met or significantly under.
