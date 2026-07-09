# One-shot setup for a fresh clone. Detects the platform (or take -Platform) and
# bootstraps *that platform's* toolchain: installs prerequisites, fetches the vendor
# SDK, gets the model + sample audio, and builds with the right backend enabled.
# Everything it pulls in is gitignored, so nothing generated pollutes the repo.
#
#   .\bootstrap.ps1                      # auto-detect platform, full setup + build
#   .\bootstrap.ps1 -Platform amd        # force the AMD (Ryzen AI) toolchain
#   .\bootstrap.ps1 -SkipPrereqs         # assume toolchain/Python already installed
#
# Per platform:
#   intel     -> setup-intel.ps1     (OpenVINO GenAI SDK) + get-model.ps1
#   amd       -> setup-amd.ps1       (Ryzen AI SDK + NPU driver) + get-amd-model.ps1
#   qualcomm  -> setup-qualcomm.ps1  (QNN SDK scaffold)
#
# Prerequisites installed via winget (if missing): Visual Studio Build Tools
# (C++ workload) and Python. A reboot may be required after a first-time VS install.
[CmdletBinding()]
param(
    [ValidateSet("auto", "intel", "amd", "qualcomm")][string]$Platform = "auto",
    [switch]$SkipPrereqs,
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$scripts = $PSScriptRoot

function Detect-Platform {
    # CPU manufacturer is the reliable signal (Win32_PnPSignedDriver name matching is
    # flaky). Each vendor's CPU implies its NPU: AMD->XDNA, Intel->Intel NPU,
    # Snapdragon->Hexagon. The real NPU device only confirms.
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Manufacturer
    switch -Regex ($cpu) {
        'AMD'          { return 'amd' }
        'Qualcomm|ARM' { return 'qualcomm' }
        'Intel'        { return 'intel' }
        default {
            $amdNpu = Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like '*NPU Compute Accelerator*' }
            if ($amdNpu) { return 'amd' }
            return 'intel'
        }
    }
}

if ($Platform -eq "auto") {
    $Platform = Detect-Platform
    Write-Host "Detected platform: $Platform" -ForegroundColor Cyan
} else {
    Write-Host "Platform (forced): $Platform" -ForegroundColor Cyan
}

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

# Per-platform: SDK/tools + model, plus the build toggles for build.ps1.
$buildArgs = @{ Configuration = $Configuration }
switch ($Platform) {
    'intel' {
        Write-Host "== Step 2/5: OpenVINO GenAI SDK ==" -ForegroundColor Cyan
        & (Join-Path $scripts "setup-intel.ps1")
        Write-Host "== Step 3/5: export model ==" -ForegroundColor Cyan
        & (Join-Path $scripts "get-model.ps1")
    }
    'amd' {
        Write-Host "== Step 2/5: Ryzen AI SDK + NPU driver ==" -ForegroundColor Cyan
        & (Join-Path $scripts "setup-amd.ps1")
        Write-Host "== Step 3/5: download AMD model ==" -ForegroundColor Cyan
        & (Join-Path $scripts "get-amd-model.ps1")
        $buildArgs.EnableAmd = $true
        $buildArgs.DisableIntel = $true
    }
    'qualcomm' {
        Write-Host "== Step 2/5: ONNX Runtime + QNN EP ==" -ForegroundColor Cyan
        & (Join-Path $scripts "setup-qualcomm.ps1")
        Write-Host "== Step 3/5: models (HF snapshot if missing) ==" -ForegroundColor Cyan
        if (-not (Test-Path (Join-Path (Split-Path $scripts -Parent) "models\whisper\en-static-onnx\encoder_model.onnx"))) {
            & (Join-Path $scripts "get-models.ps1")
        }
        $buildArgs.EnableQualcomm = $true
        $buildArgs.DisableIntel = $true
    }
}

Write-Host "== Step 4/5: sample audio ==" -ForegroundColor Cyan
& (Join-Path $scripts "get-audio.ps1")

Write-Host "== Step 5/5: build ==" -ForegroundColor Cyan
& (Join-Path $scripts "build.ps1") @buildArgs

Write-Host ""
Write-Host "Bootstrap complete ($Platform). Run it with:" -ForegroundColor Green
switch ($Platform) {
    'intel'    { Write-Host "  .\scripts\run.ps1 -Backend intel -Device npu" -ForegroundColor White }
    'amd'      { Write-Host "  .\scripts\run.ps1 -Backend amd -Device npu" -ForegroundColor White }
    'qualcomm' { Write-Host "  .\scripts\run.ps1 -Backend qualcomm -Device npu" -ForegroundColor White }
}
