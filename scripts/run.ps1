# Runs the built whisper_hal app. Runtime DLLs are copied next to the exe by the
# build, so no environment setup is required. Model/audio default to the repo-local
# copies produced by get-model.ps1 / get-audio.ps1.
#
#   .\run.ps1                                   # NPU, exported model, cache demo
#   .\run.ps1 -Device cpu -Backend intel
#   .\run.ps1 -Model <ov_dir> -Audio <wav> -NoCache

[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Audio = "",
    [ValidateSet("auto", "intel", "amd", "qualcomm")][string]$Backend = "intel",
    [ValidateSet("npu", "gpu", "cpu")][string]$Device = "npu",
    [int]$Runs = 5,
    [string]$Configuration = "Release",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
if (-not $Model) { $Model = Join-Path $root "models\whisper-tiny-en-ov" }
if (-not $Audio) { $Audio = Join-Path $root "models\jfk.wav" }

$exe = Join-Path $root "build\x64\$Configuration\WhisperNpuHal.App.exe"
if (-not (Test-Path $exe)) { Write-Host "Not built: $exe  (run .\scripts\build.ps1 or bootstrap.ps1)" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Model)) { Write-Host "Model not found: $Model  (run .\scripts\get-model.ps1)" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Audio)) { Write-Host "Audio not found: $Audio  (run .\scripts\get-audio.ps1)" -ForegroundColor Red; exit 1 }

$cacheArgs = @()
if (-not $NoCache) {
    $cacheDir = Join-Path $root "build\cache\$Device"
    $cacheArgs = @("--cache", $cacheDir)
}

& $exe $Model $Audio $Backend $Device $Runs @cacheArgs
