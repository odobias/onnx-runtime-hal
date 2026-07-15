# Builds NpuInferenceBench.sln with MSBuild. Prefers Visual Studio 2026 (v18); falls
# back to the newest installed VS if 2026 is not present. PlatformToolset is
# $(DefaultPlatformToolset), so the build tracks whichever VS is used.
#
#   .\build.ps1                       # Release, host platform, Intel backend
#   .\build.ps1 -Configuration Debug
#   .\build.ps1 -EnableAmd -DisableIntel
#   .\build.ps1 -EnableOrt -DisableIntel
#   .\build.ps1 -EnableDirectML -DisableIntel
#   .\build.ps1 -EnableWinML -DisableIntel
#   .\build.ps1 -EnableOrt -DisableIntel -OrtDir C:\onnxruntime
#   .\build.ps1 -Platform ARM64 -EnableQualcomm -DisableIntel  # native Snapdragon

[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release",
    [ValidateSet("x64", "ARM64")][string]$Platform = "",
    [switch]$EnableAmd,
    [switch]$EnableOrt,
    # Use the official Microsoft.ML.OnnxRuntime.DirectML SDK assembled by
    # tools/fetch/get-onnxruntime-directml.ps1. Emits a separate *-dml build tree.
    [switch]$EnableDirectML,
    # Windows ML self-contained ORT + dynamic EP catalog. Emits a separate
    # <platform>-winml tree and cannot share a process with bundled EP runtimes.
    [switch]$EnableWinML,
    # Intel native path via ONNX Runtime's OpenVINO EP. Makes the unified ORT binary
    # cover Intel NPU/GPU/CPU. Implies -EnableOrt and forces the OpenVINO GenAI
    # backend OFF (they need incompatible openvino.dll versions in one process).
    [switch]$EnableOvep,
    [switch]$DisableIntel,
    [switch]$EnableQualcomm,
    # Also bundle the ~155 MB AMD-native whisper/amd model into the package.
    # Off by default -- the unified auto/onnx-static path runs the portable static
    # model and does not need it; enable only to run the amd-native variant offline.
    [switch]$BundleAmdModel,
    [string]$RyzenAiDir = "",
    [string]$OrtDir = "",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

# Default the target platform to the host architecture so a native ARM64
# (Snapdragon) machine builds native ARM64 instead of x64-emulated.
if (-not $Platform) {
    $Platform = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { "ARM64" } else { "x64" }
}

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$sln = Join-Path $root "NpuInferenceBench.sln"
$vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

function Find-MSBuild {
    # Prefer VS 2026 (v18), then any latest with MSBuild + VC tools.
    $ranges = @("[18.0,19.0)", "[17.0,18.0)", "[16.0,17.0)")
    foreach ($r in $ranges) {
        $p = & $vsw -version $r -latest -products * -property installationPath 2>$null
        if ($p) {
            $mb = Join-Path $p "MSBuild\Current\Bin\amd64\MSBuild.exe"
            if (Test-Path $mb) { return @{ msbuild = $mb; vs = $p; range = $r } }
            $mb = Join-Path $p "MSBuild\Current\Bin\MSBuild.exe"
            if (Test-Path $mb) { return @{ msbuild = $mb; vs = $p; range = $r } }
        }
    }
    return $null
}

$found = Find-MSBuild
if (-not $found) { Write-Host "No Visual Studio with MSBuild found." -ForegroundColor Red; exit 1 }
$isVs2026 = $found.range -eq "[18.0,19.0)"
Write-Host "Using MSBuild: $($found.msbuild)" -ForegroundColor Cyan
Write-Host ("VS install   : {0}  {1}" -f $found.vs, ($(if ($isVs2026) { '(VS 2026)' } else { '(NOT VS 2026 -- fallback)' }))) -ForegroundColor $(if ($isVs2026) { 'Green' } else { 'Yellow' })

$target = if ($Rebuild) { "Rebuild" } else { "Build" }
if ($EnableWinML -and ($EnableDirectML -or $EnableOvep -or $EnableAmd -or $EnableQualcomm)) {
    throw "-EnableWinML cannot be combined with bundled DirectML, OVEP, AMD, or Qualcomm runtimes."
}
$packSelections = @($EnableDirectML, $EnableOvep, $EnableAmd, $EnableQualcomm) |
    Where-Object { $_ }
if (@($packSelections).Count -gt 1) {
    throw "Select only one isolated runtime pack: DirectML, OVEP, AMD, or Qualcomm."
}
if ($EnableWinML -and -not $DisableIntel) {
    Write-Host "EnableWinML: forcing Intel (OpenVINO GenAI) backend OFF -- Windows ML owns the runtime and EP catalog." -ForegroundColor Yellow
    $DisableIntel = $true
}
# OVEP and the OpenVINO GenAI backend both load openvino.dll, at versions that
# cannot coexist in one process. Selecting OVEP therefore disables Intel-GenAI.
if ($EnableOvep -and -not $DisableIntel) {
    Write-Host "EnableOvep: forcing Intel (OpenVINO GenAI) backend OFF -- OVEP replaces it (openvino.dll version conflict)." -ForegroundColor Yellow
    $DisableIntel = $true
}
$requestedOrtPack = $EnableOrt -or $EnableDirectML -or $EnableWinML -or
    $EnableOvep -or $EnableAmd -or $EnableQualcomm
if ($requestedOrtPack -and -not $DisableIntel) {
    Write-Host "ONNX Runtime pack selected: forcing Intel OpenVINO GenAI backend OFF to keep runtime DLLs isolated." -ForegroundColor Yellow
    $DisableIntel = $true
}
if ($EnableDirectML -and -not $OrtDir) {
    $directmlDir = Join-Path $root "third_party\onnxruntime-directml"
    if (-not (Test-Path (Join-Path $directmlDir "include\onnxruntime_cxx_api.h"))) {
        Write-Host "DirectML SDK missing. Run: .\tools\fetch\get-onnxruntime-directml.ps1" -ForegroundColor Red
        exit 1
    }
    $OrtDir = $directmlDir
}
if ($EnableWinML) {
    $winmlDir = Join-Path $root "third_party\windows-ml"
    if (-not (Test-Path (Join-Path $winmlDir "include\WinMLEpCatalog.h")) -or
        -not (Test-Path (Join-Path $winmlDir "bin\$Platform\onnxruntime.dll"))) {
        Write-Host "Windows ML SDK missing. Run: .\tools\setup\setup-winml.ps1 -Platform $Platform" -ForegroundColor Red
        exit 1
    }
}
$enableIntel = -not $DisableIntel
$enableOrt = $requestedOrtPack
$args = @(
    $sln,
    "/t:$target",
    "/p:Configuration=$Configuration",
    "/p:Platform=$Platform",
    "/p:EnableIntel=$([bool]$enableIntel)".ToLower(),
    "/p:EnableOrt=$([bool]$enableOrt)".ToLower(),
    "/p:EnableWinML=$([bool]$EnableWinML)".ToLower(),
    "/p:EnableOvep=$([bool]$EnableOvep)".ToLower(),
    "/p:EnableAmd=$([bool]$EnableAmd)".ToLower(),
    "/p:EnableQualcomm=$([bool]$EnableQualcomm)".ToLower(),
    "/m",
    "/nologo",
    "/v:minimal"
)
if ($BundleAmdModel) {
    $args += "/p:BundleAmdModel=true"
}
if ($RyzenAiDir) {
    $args += "/p:RyzenAiDir=$RyzenAiDir"
}
if ($OrtDir) {
    $args += "/p:OrtDir=$OrtDir"
}
if ($EnableDirectML) {
    $args += "/p:PlatformOutTag=$Platform-dml"
}
& $found.msbuild @args
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed." -ForegroundColor Red; exit 1 }

$platformOutTag = if ($EnableOvep) {
    "$Platform-ovep"
} elseif ($EnableWinML) {
    "$Platform-winml"
} elseif ($EnableDirectML) {
    "$Platform-dml"
} elseif ($EnableAmd) {
    "$Platform-amd"
} elseif ($EnableQualcomm) {
    "$Platform-qualcomm"
} else {
    $Platform
}
$exe = Join-Path $root "build\$platformOutTag\$Configuration\NpuInferenceBench.exe"
Write-Host ""
Write-Host "Built: $exe" -ForegroundColor Green
Write-Host "Run  : .\benchmark\run-whisper.ps1" -ForegroundColor Cyan
