# setup.ps1 — Windows setup script for Gazebo Swarm MARL
# Run as Administrator: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Gazebo Swarm MARL — Windows Setup ===" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check Docker Desktop ──────────────────────────────────────────────
Write-Host "[1/4] Checking Docker Desktop..." -ForegroundColor Yellow
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    Write-Host "  Docker Desktop not found." -ForegroundColor Red
    Write-Host "  Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    Write-Host "  - Enable WSL 2 backend during installation"
    Write-Host "  - After install, start Docker Desktop and run this script again."
    Write-Host ""
    Write-Host "  Attempting install via winget..." -ForegroundColor Yellow
    try {
        winget install -e --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
        Write-Host "  Docker Desktop installed. Please START Docker Desktop, then re-run this script." -ForegroundColor Green
    } catch {
        Write-Host "  winget install failed. Download manually from the URL above." -ForegroundColor Red
    }
    exit 1
}

$dockerVersion = & docker --version 2>&1
Write-Host "  $dockerVersion" -ForegroundColor Green

# Check Docker daemon is running
$dockerInfo = & docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "  Docker daemon is running." -ForegroundColor Green

# ── Step 2: Check VcXsrv (X11 server for GUI) ─────────────────────────────────
Write-Host ""
Write-Host "[2/4] Checking VcXsrv (X11 server for Gazebo GUI)..." -ForegroundColor Yellow
$vcxsrv = Get-Command vcxsrv -ErrorAction SilentlyContinue
$vcxsrvPath = "C:\Program Files\VcXsrv\vcxsrv.exe"
if (-not $vcxsrv -and -not (Test-Path $vcxsrvPath)) {
    Write-Host "  VcXsrv not found. Installing via winget..." -ForegroundColor Yellow
    try {
        winget install -e --id marha.VcXsrv --accept-package-agreements --accept-source-agreements
        Write-Host "  VcXsrv installed." -ForegroundColor Green
    } catch {
        Write-Host "  winget install failed. Download VcXsrv from:" -ForegroundColor Red
        Write-Host "    https://sourceforge.net/projects/vcxsrv/"
        Write-Host "  Install it, then re-run this script."
        exit 1
    }
} else {
    Write-Host "  VcXsrv is available." -ForegroundColor Green
}

# Launch VcXsrv if not already running
$vcxsrvProc = Get-Process vcxsrv -ErrorAction SilentlyContinue
if (-not $vcxsrvProc) {
    Write-Host "  Starting VcXsrv with -ac (no access control)..." -ForegroundColor Yellow
    $xsrvExe = if (Test-Path $vcxsrvPath) { $vcxsrvPath } else { "vcxsrv" }
    Start-Process $xsrvExe -ArgumentList ":0 -multiwindow -ac -silent-dup-error" -WindowStyle Hidden
    Start-Sleep -Seconds 2
    Write-Host "  VcXsrv started on display :0" -ForegroundColor Green
} else {
    Write-Host "  VcXsrv already running." -ForegroundColor Green
}

# ── Step 3: Build Docker image ─────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Building Docker image (this takes 20-40 min on first run)..." -ForegroundColor Yellow
Write-Host "  The image compiles PX4 SITL from source — please be patient."
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

& docker compose -f docker-compose.windows.yml build
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Docker build FAILED. Check errors above." -ForegroundColor Red
    exit 1
}
Write-Host "  Docker image built successfully." -ForegroundColor Green

# ── Step 4: Launch simulation ──────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] Starting simulation container..." -ForegroundColor Yellow
Write-Host "  Inside the container, run:  bash run.sh"
Write-Host "  Then type 'help' for simulation commands."
Write-Host ""

& docker compose -f docker-compose.windows.yml run --rm swarm bash
