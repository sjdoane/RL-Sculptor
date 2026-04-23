# WSL2 setup — from zero to `./run.sh`

This is the day-one setup for Reward Sculptor on Windows + WSL2. If
you're on native Linux, skip to step 3.

## 1. WSL2 + Ubuntu 24.04

```powershell
# PowerShell (Admin). wsl.exe --install is a one-shot.
wsl --install -d Ubuntu-24.04
```

Reboot when prompted. Ubuntu launches, prompts for a username +
password. That's your WSL user.

If `wsl --version` shows something older than 2.x, run
`wsl --set-default-version 2` and `wsl --update` before the install.

## 2. CUDA pass-through verification

WSL2 inherits the Windows host's NVIDIA driver — there is **no Linux
CUDA driver to install inside WSL**. You install only the CUDA
toolkit (headers + libs + nvcc).

Check the host side:

```bash
# Inside WSL:
nvidia-smi
```

If `nvidia-smi` prints a device table (RTX 5070 Laptop GPU, etc.),
CUDA pass-through is working. If it says "command not found" or
"NVIDIA-SMI has failed":

1. **Update your Windows NVIDIA driver** to a version supporting WSL2
   GPU (530+). Restart.
2. Verify the host driver on Windows: `nvidia-smi.exe` in PowerShell.
3. If that works but WSL's `nvidia-smi` doesn't, reinstall Ubuntu
   inside WSL and re-test.

Microsoft's canonical doc:
https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

## 3. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version  # expect 0.x.x
```

uv is what this project uses instead of pip / poetry / conda. It's
fast, handles Python version pinning, and plays nicely with WSL
filesystem.

## 4. Clone the project

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/sjdoane/RL-Sculptor.git
# The repo has two subdirs that are the two uv projects:
ls RL-Sculptor   # RewardSculptor/  reward-sculptor-ui/
```

Convention: keep these two directories next to each other. The UI's
`pyproject.toml` path-installs `../RewardSculptor` for the sculptor
library.

For the AME456 quadruped capstone (separate project): clone whichever
upstream you're using as a sibling of `RewardSculptor/`. The AME456
env imports `sculptor.reward.compute_reward` via a sys.path shim — it
expects `../RewardSculptor/` relative to the AME456 checkout.

## 5. First-time `uv sync`

```bash
cd ~/projects/RewardSculptor
uv sync                    # pulls sculptor + mjlab[cu128] + torch+cu130 etc.
# ~5-10 min on a fresh machine; most of that is torch.

cd ~/projects/reward-sculptor-ui
uv sync                    # pulls FastAPI + path-installs sculptor editable
pnpm install --dir frontend # ~1 min
```

Verify the GPU stack:

```bash
cd ~/projects/RewardSculptor
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA GeForce RTX 5070 Laptop GPU  (or your GPU)
```

## 6. Set your Anthropic API key

```bash
cp ~/projects/RewardSculptor/.env.example ~/projects/RewardSculptor/.env
$EDITOR ~/projects/RewardSculptor/.env
# Fill in:
# ANTHROPIC_API_KEY=sk-ant-api03-...
```

Live sculpt runs need this; `--dry-run` works without it. Get a key
at https://console.anthropic.com/settings/keys.

## 7. One-liner daily workflow

```bash
cd ~/projects/reward-sculptor-ui && ./run.sh
```

`run.sh` starts:
- uvicorn (backend) on :8000
- Vite (frontend) on :5173 with `/api` + `/ws` proxied to :8000

Open http://localhost:5173 in Chrome / Edge / Firefox on the Windows
host — WSL2's localhost is bridged automatically.

## Quick sanity checks

```bash
# Backend + sculptor test suites (fast):
cd ~/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q
cd ~/projects/RewardSculptor && uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py

# GPU smoke (cached; fast after first run):
cd ~/projects/RewardSculptor && uv run pytest -m gpu -q
```

## Common WSL pitfalls

- **Disk on `/mnt/c` is slow.** Keep projects under `~/projects/` (the
  ext4 filesystem), never under `/mnt/c/Users/.../Projects/`. 10-20x
  speed difference on `uv sync`.
- **Git line endings.** `git config --global core.autocrlf input` so
  Windows-style CRLF doesn't sneak in.
- **OpenGL rendering.** MuJoCo's renderer auto-picks the backend under
  WSLg. If a thumbnail render fails with
  `AttributeError: 'NoneType' object has no attribute 'glGetError'`,
  **unset** `MUJOCO_GL` (don't set it to osmesa) — the default path
  works with WSLg.
- **systemd-resolved.** If arxiv ingest hangs at DNS, restart it:
  `sudo systemctl restart systemd-resolved`.

## Minimum hardware for the mjlab path

- NVIDIA GPU with CUDA 12.4+ driver.
- ≥ 6 GiB VRAM for quadrupeds; ≥ 8 GiB for G1 at num_envs=1024.
- 32 GiB disk for the dev envs + Menagerie cache.
- WSL2 running Ubuntu 22.04 or 24.04.

The gym_sb3 path works on anything that runs Python 3.10 + torch.
