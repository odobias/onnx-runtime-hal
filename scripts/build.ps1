# Builds WhisperNpuHal.sln with MSBuild. Prefers Visual Studio 2026 (v18); falls
# back to the newest installed VS if 2026 is not present. PlatformToolset is
# $(DefaultPlatformToolset), so the build tracks whichever VS is used.
#
#   .\build.ps1                       # Release|x64, Intel backend
#   .\build.ps1 -Configuration Debug
#   .\build.ps1 -EnableAmd -DisableIntel
#   .\build.ps1 -EnableOrt -DisableIntel
#   .\build.ps1 -EnableOrt -DisableIntel -OrtDir C:\onnxruntime

[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release",
    [switch]$EnableAmd,
    [switch]$EnableOrt,
    [switch]$DisableIntel,
    [switch]$EnableQualcomm,
    [string]$RyzenAiDir = "",
    [string]$OrtDir = "",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
$sln = Join-Path $root "WhisperNpuHal.sln"
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
$enableIntel = -not $DisableIntel
$enableOrt = $EnableOrt -or $EnableAmd
$args = @(
    $sln,
    "/t:$target",
    "/p:Configuration=$Configuration",
    "/p:Platform=x64",
    "/p:EnableIntel=$([bool]$enableIntel)".ToLower(),
    "/p:EnableOrt=$([bool]$enableOrt)".ToLower(),
    "/p:EnableAmd=$([bool]$EnableAmd)".ToLower(),
    "/p:EnableQualcomm=$([bool]$EnableQualcomm)".ToLower(),
    "/m",
    "/nologo",
    "/v:minimal"
)
if ($RyzenAiDir) {
    $args += "/p:RyzenAiDir=$RyzenAiDir"
}
if ($OrtDir) {
    $args += "/p:OrtDir=$OrtDir"
}
& $found.msbuild @args
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed." -ForegroundColor Red; exit 1 }

$exe = Join-Path $root "build\x64\$Configuration\WhisperNpuHal.App.exe"
Write-Host ""
Write-Host "Built: $exe" -ForegroundColor Green
Write-Host "Run  : .\scripts\run.ps1" -ForegroundColor Cyan
