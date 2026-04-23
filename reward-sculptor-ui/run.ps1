# Reward Sculptor UI — one-command start (Windows PowerShell).
#
# Starts the FastAPI backend + Vite dev server, opens the UI in your
# default browser, and cleans both up on Ctrl+C.
#
# Usage (from the repo root):
#   pwsh ./run.ps1
#   # or with a custom backend port
#   $env:RS_PORT = 8765; pwsh ./run.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$BackendPort = if ($env:RS_PORT) { [int]$env:RS_PORT } else { 8000 }
$FrontendPort = 5173

function Write-Err($msg) { Write-Host "[run] $msg" -ForegroundColor Red }
function Write-Info($msg) { Write-Host "[run] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[run] $msg" -ForegroundColor Yellow }

# ── dependency checks ────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Err "uv is not on PATH. Install from https://docs.astral.sh/uv/"
    exit 1
}
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    Write-Err "pnpm is not on PATH. Install with: npm i -g pnpm"
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Err ".venv not found. Run 'uv sync' first, then re-run this script."
    exit 1
}
if (-not (Test-Path "frontend/node_modules")) {
    Write-Err "frontend/node_modules not found. Run 'pnpm install --dir frontend' first."
    exit 1
}

# ── port check ───────────────────────────────────────────────────────
function Test-PortFree($port) {
    try {
        $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        return -not $c
    } catch {
        return $true
    }
}

if (-not (Test-PortFree $BackendPort)) {
    Write-Err ("Port {0} is in use — set `$env:RS_PORT or free it" -f $BackendPort)
    exit 1
}
if (-not (Test-PortFree $FrontendPort)) {
    Write-Err "Port 5173 is in use — close the other Vite dev server or use a different workspace."
    exit 1
}

# ── API key warning (non-fatal) ──────────────────────────────────────
$envFile = Join-Path $PSScriptRoot "../RewardSculptor/.env"
$apiKey = $env:ANTHROPIC_API_KEY
if (-not $apiKey -and (Test-Path $envFile)) {
    # Pick out ANTHROPIC_API_KEY=... from ../RewardSculptor/.env without
    # sourcing the whole file.
    $match = Select-String -Path $envFile -Pattern '^ANTHROPIC_API_KEY=(.+)$' -ErrorAction SilentlyContinue
    if ($match) {
        $apiKey = $match.Matches.Groups[1].Value.Trim('"').Trim("'")
    }
}
if (-not $apiKey) {
    Write-Warn "ANTHROPIC_API_KEY not set — live runs will fail, dry-run only."
}

# ── spawn both subprocesses ──────────────────────────────────────────
$backendArgs = @(
    "run", "uvicorn", "backend.main:app",
    "--host", "127.0.0.1", "--port", "$BackendPort",
    "--log-level", "info"
)
$env:UV_LINK_MODE = "copy"

Write-Info "starting backend on http://127.0.0.1:$BackendPort"
$backend = Start-Process -FilePath "uv" -ArgumentList $backendArgs `
    -PassThru -NoNewWindow

# Wait briefly for the backend to listen — prevents "connection refused"
# races when Vite's proxy fires immediately.
$backendReady = $false
foreach ($_ in 1..30) {
    Start-Sleep -Milliseconds 300
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($r.StatusCode -eq 200) { $backendReady = $true; break }
    } catch { }
}
if (-not $backendReady) {
    Write-Err "backend did not respond on /health within 10s. Is it crashing? See stderr."
    if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    exit 1
}

Write-Info "starting frontend on http://127.0.0.1:$FrontendPort"
# pnpm via Start-Process doesn't resolve .cmd shims — call pnpm.cmd
# explicitly so we always start the same binary we tested with.
$pnpmCmd = (Get-Command pnpm).Source
$frontend = Start-Process -FilePath $pnpmCmd -ArgumentList @("--dir", "frontend", "dev") `
    -PassThru -NoNewWindow

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:$FrontendPort"

Write-Info "both processes running. Ctrl+C to stop."

# ── Ctrl+C handler ───────────────────────────────────────────────────
function Stop-Both {
    Write-Info "shutting down…"
    foreach ($p in @($backend, $frontend)) {
        if ($p -and -not $p.HasExited) {
            try {
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            } catch { }
        }
    }
}
try {
    while (-not $backend.HasExited -and -not $frontend.HasExited) {
        Start-Sleep -Seconds 1
    }
} finally {
    Stop-Both
}
