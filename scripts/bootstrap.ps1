# One-shot setup for a fresh clone: installs prerequisites, fetches the OpenVINO
# SDK, exports the model, downloads sample audio, and builds. Everything it pulls in
# is gitignored, so nothing generated pollutes the repo.
#
#   .\bootstrap.ps1                 # full setup + build
#   .\bootstrap.ps1 -SkipPrereqs    # assume toolchain/Python already installed
#
# Prerequisites installed via winget (if missing): Visual Studio Build Tools
# (C++ workload) and Python. A reboot may be required after a first-time VS install.
[CmdletBinding()]
param(
    [switch]$SkipPrereqs,
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$scripts = $PSScriptRoot

function Have-MSBuild {
    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsw)) { return $false }
    $p = & $vsw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    return [bool]$p
}

if (-not $SkipPrereqs) {
    Write-Host "== Step 1/5: prerequisites ==" -ForegroundColor Cyan

    if (Have-MSBuild) {
        Write-Host "  Visual Studio C++ toolchain: found" -ForegroundColor Green
    } else {
        Write-Host "  Installing Visual Studio Build Tools (C++ workload) via winget..." -ForegroundColor Yellow
        winget install --id Microsoft.VisualStudio.BuildTools -e --accept-source-agreements --accept-package-agreements `
            --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        if (-not (Have-MSBuild)) {
            Write-Host "  VS Build Tools install may need a reboot. Reboot, then re-run bootstrap.ps1." -ForegroundColor Yellow
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Host "  Python: found" -ForegroundColor Green
    } else {
        Write-Host "  Installing Python via winget..." -ForegroundColor Yellow
        winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
        Write-Host "  Python installed; you may need a new shell for PATH to update." -ForegroundColor Yellow
    }
} else {
    Write-Host "== Step 1/5: prerequisites (skipped) ==" -ForegroundColor DarkGray
}

Write-Host "== Step 2/5: OpenVINO GenAI SDK ==" -ForegroundColor Cyan
& (Join-Path $scripts "setup-intel.ps1")

Write-Host "== Step 3/5: export model ==" -ForegroundColor Cyan
& (Join-Path $scripts "get-model.ps1")

Write-Host "== Step 4/5: sample audio ==" -ForegroundColor Cyan
& (Join-Path $scripts "get-audio.ps1")

Write-Host "== Step 5/5: build ==" -ForegroundColor Cyan
& (Join-Path $scripts "build.ps1") -Configuration $Configuration

Write-Host ""
Write-Host "Bootstrap complete. Run it with:" -ForegroundColor Green
Write-Host "  .\scripts\run.ps1" -ForegroundColor White
