#!/usr/bin/env bash
# Reward Sculptor UI — one-command start (POSIX / macOS / Linux).
#
# Starts the FastAPI backend + Vite dev server, opens the UI in the
# default browser, and cleans both up on Ctrl+C.
#
# Usage (from the repo root):
#   ./run.sh
#   # or with a custom backend port
#   RS_PORT=8765 ./run.sh

set -euo pipefail

cd "$(dirname "$0")"

BACKEND_PORT="${RS_PORT:-8000}"
FRONTEND_PORT="5173"

log()  { printf '[run] %s\n' "$*"; }
warn() { printf '[run] %s\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

# ── dependency checks ────────────────────────────────────────────────
command -v uv   >/dev/null 2>&1 || die "uv is not on PATH. Install from https://docs.astral.sh/uv/"
command -v pnpm >/dev/null 2>&1 || die "pnpm is not on PATH. Install with: npm i -g pnpm"

[ -d ".venv" ] || die ".venv not found. Run 'uv sync' first, then re-run this script."
[ -d "frontend/node_modules" ] || die "frontend/node_modules not found. Run 'pnpm install --dir frontend' first."

# ── port check ───────────────────────────────────────────────────────
port_free() {
    local p="$1"
    if command -v lsof >/dev/null 2>&1; then
        ! lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v ss >/dev/null 2>&1; then
        ! ss -ltn "sport = :$p" 2>/dev/null | tail -n +2 | grep -q ":$p"
    else
        # no usable probe — assume free
        return 0
    fi
}

# §Ship-8c: auto-kill orphaned uvicorn / vite processes that held the
# port after a prior `./run.sh` exited without cleanup (pkill on the
# parent shell, crash, system sleep, etc.). Only targets processes
# whose command name is our own uvicorn / vite invocations — any other
# app on the port is left alone so we don't nuke a user's unrelated
# service. Per Sam's rule: no terminal commands besides `./run.sh`.
pids_holding_port() {
    local p="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$p" -sTCP:LISTEN -t 2>/dev/null | sort -u
    elif command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :$p" 2>/dev/null \
            | awk -F'pid=' '/pid=/ {for (i=2; i<=NF; i++) { split($i, a, ","); print a[1] }}' \
            | sort -u
    else
        # §Ship-8c (critique 6): neither probe available. Caller turns
        # "no owner found" into a confusing uvicorn bind error on
        # retry; emit a clear signal up the stack instead.
        echo "__NOPROBE__"
    fi
}

# §Ship-8c hotfix (critique 5): match must require BOTH our spawn
# signature AND a token identifying our app (backend.main or our vite
# entry). Plain `multiprocessing.spawn` matched any Python multi-proc
# child — too broad.
is_our_orphan() {
    local pid="$1"
    local cmd=""
    [ -r "/proc/$pid/cmdline" ] && cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    [ -z "$cmd" ] && cmd="$(ps -p "$pid" -o cmd= 2>/dev/null || true)"
    # Top-level patterns.
    case "$cmd" in
        *"uvicorn"*"backend.main"*)        return 0 ;;
        *"node_modules/.bin/vite"*)        return 0 ;;
        *"vite"*"vite.js"*)                return 0 ;;
    esac
    # multiprocessing reload child — only count if its parent lineage
    # has our uvicorn. Walk up via /proc/<pid>/status's PPid until we
    # find one, or hit PPid=1 (init) without finding a match.
    case "$cmd" in
        *"multiprocessing.spawn"*"spawn_main"*)
            local ancestor_pid="$pid"
            local hops=0
            while [ "$hops" -lt 6 ]; do
                local ppid=""
                [ -r "/proc/$ancestor_pid/status" ] \
                    && ppid="$(awk '/^PPid:/ {print $2}' "/proc/$ancestor_pid/status" 2>/dev/null)"
                [ -z "$ppid" ] || [ "$ppid" -le 1 ] && break
                local parent_cmd=""
                [ -r "/proc/$ppid/cmdline" ] \
                    && parent_cmd="$(tr '\0' ' ' < "/proc/$ppid/cmdline")"
                case "$parent_cmd" in
                    *"uvicorn"*"backend.main"*) return 0 ;;
                esac
                ancestor_pid="$ppid"
                hops=$((hops + 1))
            done
            return 1
            ;;
    esac
    return 1
}

reclaim_port_if_orphan() {
    local p="$1" label="$2"
    local pid
    for pid in $(pids_holding_port "$p"); do
        if [ "$pid" = "__NOPROBE__" ]; then
            die "Port $p is in use but neither lsof nor ss is installed — can't identify the owner. Install one with 'sudo apt install lsof' (or 'iproute2' for ss), then re-run."
        fi
        if is_our_orphan "$pid"; then
            warn "port $p held by orphan $label (pid=$pid) — killing"
            kill -KILL "$pid" 2>/dev/null || true
        else
            local cmd="$(ps -p "$pid" -o cmd= 2>/dev/null || echo '<unknown>')"
            die "Port $p is in use by an unrelated process (pid=$pid: $cmd) — free it manually or set RS_PORT."
        fi
    done
    sleep 0.5  # give the kernel a beat to release the port
}

if ! port_free "$BACKEND_PORT"; then
    reclaim_port_if_orphan "$BACKEND_PORT" "uvicorn"
    port_free "$BACKEND_PORT" || die "Port $BACKEND_PORT still in use after reclaim attempt."
fi
if ! port_free "$FRONTEND_PORT"; then
    reclaim_port_if_orphan "$FRONTEND_PORT" "vite"
    port_free "$FRONTEND_PORT" || die "Port $FRONTEND_PORT still in use after reclaim attempt."
fi

# ── API key warning (non-fatal) ──────────────────────────────────────
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    if [ -f "../RewardSculptor/.env" ] && grep -qE '^ANTHROPIC_API_KEY=' "../RewardSculptor/.env"; then
        : # sculptor's .env will be picked up by its own loader
    else
        warn "ANTHROPIC_API_KEY not set — live runs will fail, dry-run only."
    fi
fi

# ── spawn + cleanup trap ─────────────────────────────────────────────
export UV_LINK_MODE=copy

# Pin MuJoCo to the headless EGL backend so the preview endpoint's
# `mujoco.Renderer` doesn't try to open a GLFW / X display on WSL2.
# Without this, the auto-picked backend can hang instead of erroring
# cleanly, wedging the /projects/{slug}/preview request and any UI
# code that awaits it. Override with `MUJOCO_GL=osmesa ./run.sh` if
# EGL isn't available on this host (software renderer, slower but
# universal).
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cleanup() {
    local status=$?
    log "shutting down…"
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup INT TERM EXIT

log "starting backend on http://127.0.0.1:$BACKEND_PORT (with --reload)"
# --reload picks up edits to both the backend package AND the editable-
# installed sculptor library, so library-side changes no longer require
# a manual backend restart during development.
uv run uvicorn backend.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" --log-level info \
    --reload \
    --reload-dir backend \
    --reload-dir ../RewardSculptor/sculptor &
BACKEND_PID=$!

# Wait for /health to respond (up to ~10 s) so Vite doesn't race
# against a half-started backend.
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        die "backend exited during startup. Check stderr above."
    fi
    sleep 0.3
done
curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1 \
    || die "backend did not respond on /health within 10s."

log "starting frontend on http://127.0.0.1:$FRONTEND_PORT"
( cd frontend && pnpm dev ) &
FRONTEND_PID=$!

# Give Vite a moment to bind, then open the browser.
sleep 2
if   command -v open     >/dev/null 2>&1; then open     "http://127.0.0.1:$FRONTEND_PORT"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://127.0.0.1:$FRONTEND_PORT"
elif command -v wslview  >/dev/null 2>&1; then wslview  "http://127.0.0.1:$FRONTEND_PORT"
fi

log "both processes running (backend pid=$BACKEND_PID, frontend pid=$FRONTEND_PID). Ctrl+C to stop."

# Block until either child exits.
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
