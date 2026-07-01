# Runs the built whisper_hal app. Runtime DLLs are copied next to the exe by the
# build, so no environment setup is required.
#
#   .\run.ps1                                   # NPU, hybrid model, cache demo on
#   .\run.ps1 -Device cpu -Backend intel
#   .\run.ps1 -Model <ov_dir> -Audio <wav> -NoCache

[CmdletBinding()]
param(
    [string]$Model = "C:\Projects\whisper-npu\whisper-tiny-en-hybrid-ov",
    [string]$Audio = "C:\Projects\whisper-npu\audio\jfk.wav",
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
$exe = Join-Path $root "build\x64\$Configuration\WhisperNpuHal.App.exe"
if (-not (Test-Path $exe)) { Write-Host "Not built: $exe  (run .\scripts\build.ps1)" -ForegroundColor Red; exit 1 }

$cacheArgs = @()
if (-not $NoCache) {
    $cacheDir = Join-Path $root "build\cache\$Device"
    $cacheArgs = @("--cache", $cacheDir)
}

& $exe $Model $Audio $Backend $Device $Runs @cacheArgs
