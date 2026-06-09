#!/usr/bin/env bash
# scripts/provision_remote.sh — provision a rented GPU pod for sculptor
# remote dispatch (§Ship 23). Idempotent: safe to re-run on the same pod.
#
# Usage:
#   ./scripts/provision_remote.sh root@HOST [-p PORT] [-i SSH_KEY] [-w WORKDIR]
#
#   -w WORKDIR   remote workdir (default ~/.sculptor_remote). On a RunPod
#                network volume use -w /workspace/sculptor_remote so the
#                venv + mirrored checkpoints survive pod restarts.
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
TORCH_SPEC="torch>=2.11.0"
MJLAB_SPEC="mjlab[cu128]>=1.3.0"
if command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
  LOCAL_TORCH=$(uv run --no-sync python -c \
    "import importlib.metadata as m; print(m.version('torch'))" 2>/dev/null || true)
  LOCAL_MJLAB=$(uv run --no-sync python -c \
    "import importlib.metadata as m; print(m.version('mjlab'))" 2>/dev/null || true)
  [ -n "${LOCAL_TORCH:-}" ] && TORCH_SPEC="torch==${LOCAL_TORCH}"
  [ -n "${LOCAL_MJLAB:-}" ] && MJLAB_SPEC="mjlab[cu128]==${LOCAL_MJLAB}"
fi
case "$TORCH_SPEC" in
  *">="*) echo "WARNING: could not detect local torch/mjlab versions (run from" >&2
          echo "         RewardSculptor/ with uv on PATH) — installing pyproject" >&2
          echo "         minimums; \`sculpt remote doctor\` may report skew." >&2 ;;
esac
echo "==> pinning: $TORCH_SPEC  $MJLAB_SPEC  (workdir: $WORKDIR)"

# ── provision over a single ssh session ─────────────────────────────
# printf %q re-quotes the args for the REMOTE shell — ssh flattens its
# argv into one string, and unquoted '>=' specs would become stdout
# redirections on the pod (installing unpinned latest + eating output).
ssh "${SSH_OPTS[@]}" "$TARGET" \
  "bash -s -- $(printf '%q ' "$TORCH_SPEC" "$MJLAB_SPEC" "$WORKDIR")" <<'REMOTE'
set -euo pipefail
TORCH_SPEC="$1"
MJLAB_SPEC="$2"
WORKDIR="$3"
case "$WORKDIR" in
  "~") WORKDIR="$HOME" ;;
  "~/"*) WORKDIR="$HOME/${WORKDIR#\~/}" ;;
esac
VENV="$WORKDIR/venv"

echo "--> [pod] workdir: $WORKDIR"
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

if [ ! -x "$HOME/.local/bin/uv" ] && ! command -v uv >/dev/null 2>&1; then
  echo "--> [pod] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$VENV/bin/python" ]; then
  echo "--> [pod] creating venv (python 3.13)"
  uv venv --python 3.13 "$VENV"
fi

echo "--> [pod] installing $TORCH_SPEC $MJLAB_SPEC imageio-ffmpeg (idempotent)"
uv pip install --python "$VENV/bin/python" -q \
  "$TORCH_SPEC" "$MJLAB_SPEC" "imageio-ffmpeg>=0.6.0"

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
remote_python = "$WORKDIR/venv/bin/python"
# rollout_remote = false   # rollouts stay local by default
EOF
