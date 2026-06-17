#!/usr/bin/env bash
# scripts/provision_remote.sh — provision a rented GPU pod for sculptor
# remote dispatch (§Ship 23). Idempotent: safe to re-run on the same pod.
#
# Usage:
#   ./scripts/provision_remote.sh root@HOST [-p PORT] [-i SSH_KEY] [-w WORKDIR]
#
#   -w WORKDIR   remote workdir for mirror/code/caches (default
#                ~/.sculptor_remote). On a RunPod network volume use
#                -w /workspace/sculptor_remote so mirrored checkpoints +
#                the uv wheel cache survive pod restarts.
#
# The VENV deliberately lives on pod-LOCAL disk ($HOME/.sculptor_venv),
# NOT the network volume: RunPod's mfs does not page-cache, so a
# volume-resident venv costs ~60s of import I/O (torch alone ~26-39s)
# in EVERY runner subprocess — measured live, it turned a 29s train into
# a 199s job. The trade: after a pod restart the venv is gone and this
# script must re-run — with the uv cache on the volume that re-install
# is wheel-download-free (~1-2 min).
#
# Run from inside WSL, from the RewardSculptor/ directory (so the local
# torch/mjlab versions can be detected and pinned on the remote — this
# is what keeps `sculpt remote doctor`'s version-skew check green).
#
# What it does on the pod:
#   1. ensures rsync is installed (apt),
#   2. installs uv (if missing),
#   3. creates a Python 3.13 venv at <WORKDIR>/venv,
#   4. installs torch / mjlab[cu128] / imageio-ffmpeg pinned to the
#      LOCAL versions when detectable (else the pyproject minimums),
#   5. sanity-checks nvidia-smi (driver >= R570 for Blackwell/5090),
#   6. prints the `[remote]` config block to paste into config.toml.
#
# The sculptor package itself is NOT installed here — the executor
# rsyncs the live source to <WORKDIR>/code on every dispatch and points
# PYTHONPATH at it, so local/remote sculptor skew is structurally
# impossible.

set -euo pipefail

usage() { echo "usage: $0 user@host [-p PORT] [-i SSH_KEY] [-w WORKDIR]" >&2; exit 2; }

TARGET="${1:-}"
case "$TARGET" in
  ""|-*) usage ;;            # options must come AFTER user@host
esac
case "$TARGET" in
  *@*) : ;;
  *) echo "error: target '$TARGET' must be user@host (e.g. root@1.2.3.4)" >&2; exit 2 ;;
esac
shift

PORT=22
KEY=""
WORKDIR="~/.sculptor_remote"
while getopts "p:i:w:" opt; do
  case "$opt" in
    p) PORT="$OPTARG" ;;
    i) KEY="$OPTARG" ;;
    w) WORKDIR="$OPTARG" ;;
    *) usage ;;
  esac
done

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$PORT")
[ -n "$KEY" ] && SSH_OPTS+=(-i "$KEY")

# ── detect local versions to pin (best-effort) ──────────────────────
# Pin the ENTIRE GPU stack, not just torch/mjlab: every '>=', and every
# undeclared transitive (warp-lang API drift, mujoco-warp kwargs, mjlab's
# undeclared scipy import) has bitten on a real pod. Live-pod debugging
# log: torch>=2.11 resolved 2.12.0+cu130; warp-lang 1.14 broke mjlab's
# `wp.context` access; mujoco-warp newer-than-local used a tile_cholesky
# kwarg local warp 1.12.1 lacks; scipy/wandb weren't installed at all.
# uv lives in ~/.local/bin which non-login shells may not have on PATH.
export PATH="$HOME/.local/bin:$PATH"
PIN_PKGS="torch mjlab warp-lang mujoco mujoco-warp rsl-rl-lib numpy scipy wandb imageio imageio-ffmpeg"
PINNED_SPECS=""
PY_VERSION="3.13.5"
if command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
  PINNED_SPECS=$(uv run --no-sync python -c "
import importlib.metadata as m
specs = []
for p in '$PIN_PKGS'.split():
    try:
        specs.append(f'{p}=={m.version(p)}')
    except Exception:
        pass
print(' '.join(specs))" 2>/dev/null || true)
  LOCAL_PY=$(uv run --no-sync python -c \
    "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>/dev/null || true)
  [ -n "${LOCAL_PY:-}" ] && PY_VERSION="$LOCAL_PY"
fi
if [ -z "$PINNED_SPECS" ]; then
  echo "WARNING: could not detect local package versions (run from" >&2
  echo "         RewardSculptor/ with uv on PATH) — installing pyproject" >&2
  echo "         minimums; \`sculpt remote doctor\` WILL report skew and" >&2
  echo "         API drift in warp/mujoco-warp may break training." >&2
  PINNED_SPECS="torch>=2.11.0 mjlab[cu128]>=1.3.0 imageio-ffmpeg>=0.6.0 scipy wandb"
fi
echo "==> pinning: $PINNED_SPECS"
echo "==> python $PY_VERSION  (workdir: $WORKDIR)"

# ── provision over a single ssh session ─────────────────────────────
# printf %q re-quotes the args for the REMOTE shell — ssh flattens its
# argv into one string, and unquoted '>=' specs would become stdout
# redirections on the pod (installing unpinned latest + eating output).
ssh "${SSH_OPTS[@]}" "$TARGET" \
  "bash -s -- $(printf '%q ' "$PINNED_SPECS" "$WORKDIR" "$PY_VERSION")" <<'REMOTE'
set -euo pipefail
PINNED_SPECS="$1"
WORKDIR="$2"
PY_VERSION="$3"
case "$WORKDIR" in
  "~") WORKDIR="$HOME" ;;
  "~/"*) WORKDIR="$HOME/${WORKDIR#\~/}" ;;
esac
# Venv on pod-LOCAL disk (network-fs imports cost ~60s/process); wheel
# + python download caches on the workdir so a post-restart re-provision
# is download-free.
VENV="$HOME/.sculptor_venv"
export UV_CACHE_DIR="$WORKDIR/uv_cache"

echo "--> [pod] workdir: $WORKDIR  venv: $VENV"
mkdir -p "$WORKDIR"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else
    echo "!! [pod] not root and no sudo — cannot apt-get install rsync" >&2
  fi
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "--> [pod] installing rsync"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq rsync
fi

# §Ship 32a: headless rendering — MuJoCo's offscreen renderer under
# MUJOCO_GL=egl needs the glvnd EGL front-end (libEGL.so.1). The
# nvidia driver ships libEGL_nvidia + the vendor ICD json, but pod
# images routinely lack the dispatcher; without it every remote
# rollout dies with "an OpenGL platform library has not been loaded"
# (caught live: E4 campaign first jobs, 2026-06-11).
if [ ! -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]; then
  echo "--> [pod] installing EGL/GL runtime (libegl1 libgl1 libgles2)"
  $SUDO apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
      libegl1 libgl1 libgles2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! [pod] nvidia-smi not found — is this a GPU pod?" >&2
  exit 3
fi
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "--> [pod] GPU: $GPU (driver $DRIVER)"
MAJOR=${DRIVER%%.*}
case "$MAJOR" in
  ''|*[!0-9]*)
    echo "!! [pod] cannot parse driver version '$DRIVER' from nvidia-smi" >&2
    exit 3 ;;
esac
if [ "$MAJOR" -lt 570 ]; then
  echo "!! [pod] driver $DRIVER < R570 — Blackwell (sm_120) needs >= 570; pick another pod image" >&2
  exit 3
fi

# Always use our own up-to-date uv in ~/.local/bin — pod images often
# preinstall an old system uv (seen: 0.9.0, too old to know current
# CPython patch releases, and `uv self update` is blocked for it).
if [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "--> [pod] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
echo "--> [pod] uv: $(uv -V) at $(command -v uv)"

# Managed CPython on local disk too (stdlib imports must be fast). A
# system python3.13 is NOT used: its patch version can differ from
# local (e.g. 3.13.8 broke torch 2.11 imports via a CPython inspect
# regression; local 3.13.5 is fine) — pin the exact local interpreter.
export UV_PYTHON_INSTALL_DIR="$HOME/.sculptor_pythons"
if [ ! -x "$VENV/bin/python" ]; then
  echo "--> [pod] creating venv (python $PY_VERSION)"
  uv venv --python "$PY_VERSION" "$VENV"
fi

echo "--> [pod] installing pinned stack (idempotent): $PINNED_SPECS"
# Word-splitting of $PINNED_SPECS is intentional — one spec per word.
# shellcheck disable=SC2086
uv pip install --python "$VENV/bin/python" -q $PINNED_SPECS

echo "--> [pod] torch/cuda sanity check"
"$VENV/bin/python" - <<'PY'
import json
import torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
PY
echo "--> [pod] provision complete"
REMOTE

cat <<EOF

==> provisioned. Paste into your project's config.toml (top level, NOT
    inside [adapter].config), then verify with:
    uv run sculpt remote doctor --config <config.toml>

[remote]
enabled = true
host = "${TARGET#*@}"
port = $PORT
user = "${TARGET%%@*}"
$( [ -n "$KEY" ] && echo "key_path = \"$KEY\"" || echo "# key_path = \"~/.ssh/id_ed25519\"" )
remote_workdir = "$WORKDIR"
remote_python = "~/.sculptor_venv/bin/python"
# rollout_remote = false   # rollouts stay local by default

# NOTE: after a pod restart the venv (pod-local disk) is gone — re-run
# this script (fast: wheel cache lives on the workdir) and update
# host/port here or in the UI's Settings -> Remote GPU card.
EOF
