# engram installer script
# Usage: irm <URL> | iex
#        or: .\install.ps1
#        or: .\install.ps1 -Source "C:\path\to\engram"
#
# Compatible with Windows PowerShell 5.1

param(
    [string]$Source = "git+https://github.com/ricoaiproject-cmd/engram-global.git"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " engram installer"
Write-Host "========================================"
Write-Host ""

# ----------------------------------------------------------------
# Step 1: check for / install uv
# ----------------------------------------------------------------
Write-Host "[1/4] Checking for uv..."

$uvPath = $null
try {
    $uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
} catch {}

if ($uvPath) {
    Write-Host "  uv is already installed: $uvPath"
} else {
    Write-Host "  uv not found. Installing..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Write-Host ""
        Write-Host "[ERROR] Failed to install uv."
        Write-Host "  Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }

    # Make uv visible in this session's PATH
    $uvBin = Join-Path $env:USERPROFILE ".local\bin"
    if ($env:PATH -notlike "*$uvBin*") {
        $env:PATH = "$uvBin;$env:PATH"
    }

    # Re-check
    try {
        $uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
    } catch {}

    if (-not $uvPath) {
        Write-Host ""
        Write-Host "[ERROR] uv command still not found after installation."
        Write-Host "  Restart your terminal and try again."
        exit 1
    }
    Write-Host "  uv installed: $uvPath"
}

Write-Host ""

# ----------------------------------------------------------------
# Step 2: check for / install git (needed to fetch git+ sources)
# ----------------------------------------------------------------
Write-Host "[2/4] Checking for git..."

if ($Source -like "git+*") {
    $gitPath = $null
    try {
        $gitPath = (Get-Command git -ErrorAction SilentlyContinue).Source
    } catch {}

    if ($gitPath) {
        Write-Host "  git is already installed: $gitPath"
    } else {
        Write-Host "  git not found. Installing..."
        winget install --id Git.Git -e --silent --accept-source-agreements --accept-package-agreements
        if (-not $?) {
            Write-Host ""
            Write-Host "[ERROR] Failed to install git."
            Write-Host "  Please install it manually: https://git-scm.com/downloads/win"
            exit 1
        }

        # Make git visible in this session's PATH
        $gitBin = "C:\Program Files\Git\cmd"
        if ((Test-Path $gitBin) -and ($env:PATH -notlike "*$gitBin*")) {
            $env:PATH = "$gitBin;$env:PATH"
        }

        try {
            $gitPath = (Get-Command git -ErrorAction SilentlyContinue).Source
        } catch {}

        if (-not $gitPath) {
            Write-Host ""
            Write-Host "[ERROR] git command still not found after installation."
            Write-Host "  Restart your terminal and try again."
            exit 1
        }
        Write-Host "  git installed: $gitPath"
    }
} else {
    Write-Host "  Local source — git not required (skipped)."
}

Write-Host ""

# ----------------------------------------------------------------
# Step 3: install engram
# ----------------------------------------------------------------
# On corporate/school PCs, AppData (Roaming) can live on a network server,
# and Python placed there cannot load DLLs over the network — it simply
# won't run (confirmed on real hardware). In that case, redirect uv's
# storage location to the local disk.
$appData = [Environment]::GetFolderPath("ApplicationData")
$isNetworkAppData = $false
if ($appData -like "\\*") {
    $isNetworkAppData = $true
} else {
    try {
        $drive = (Get-Item $appData -ErrorAction SilentlyContinue).PSDrive
        if ($drive -and $drive.DisplayRoot -like "\\*") {
            $isNetworkAppData = $true
        }
    } catch {}
}

if ($isNetworkAppData) {
    Write-Host "  Detected a PC where AppData is on a network location."
    Write-Host "  Redirecting Python's storage location to the local disk (required for it to work)."
    $localUv = Join-Path $env:LOCALAPPDATA "uv"
    $env:UV_TOOL_DIR = Join-Path $localUv "tools"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $localUv "python"
    # Persist this so future sessions (e.g. uv tool upgrade) use the same location
    setx UV_TOOL_DIR $env:UV_TOOL_DIR | Out-Null
    setx UV_PYTHON_INSTALL_DIR $env:UV_PYTHON_INSTALL_DIR | Out-Null
    Write-Host ""
}

Write-Host "[3/4] Installing engram..."
Write-Host "  Source: $Source"
Write-Host "  (The first run may download Python 3.12.)"
Write-Host ""

uv tool install --python 3.12 --force $Source

if (-not $?) {
    Write-Host ""
    Write-Host "[ERROR] Failed to install engram."
    Write-Host "  - Check the Source value: $Source"
    Write-Host "  - Check your network connection"
    exit 1
}

# Add the uv tool shim directory to PATH
$uvToolBin = Join-Path $env:USERPROFILE ".local\bin"
if ($env:PATH -notlike "*$uvToolBin*") {
    $env:PATH = "$uvToolBin;$env:PATH"
}

Write-Host ""
Write-Host "  engram installed."
Write-Host ""

# ----------------------------------------------------------------
# Step 4: run the setup wizard
# ----------------------------------------------------------------
Write-Host "[4/4] Running the setup wizard..."
Write-Host ""

$engramExe = Join-Path $env:USERPROFILE ".local\bin\engram.exe"
if (-not (Test-Path $engramExe)) {
    # Fallback: look it up on PATH
    try {
        $engramExe = (Get-Command engram -ErrorAction SilentlyContinue).Source
    } catch {}
}

if (-not $engramExe) {
    Write-Host "[ERROR] engram command not found."
    Write-Host "  Restart your terminal and run 'engram setup'."
    exit 1
}

& $engramExe setup

if (-not $?) {
    Write-Host ""
    Write-Host "[ERROR] The setup wizard failed."
    Write-Host "  Fix the issue, then re-run 'engram setup'."
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host " Installation complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  - Restart your agent (e.g. Claude Code)"
Write-Host "  - Verify with: engram doctor"
Write-Host ""
