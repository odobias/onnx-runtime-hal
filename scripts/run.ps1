# Runs the built whisper_hal app. Runtime DLLs are copied next to the exe by the
# build, so no environment setup is required. Model/audio default to the repo-local
# copies produced by get-model.ps1 / get-audio.ps1.
#
#   .\run.ps1                                   # NPU, Intel exported model, cache demo
#   .\run.ps1 -Device cpu -Backend intel
#   .\run.ps1 -Device cpu -Threads 8            # pin OpenVINO CPU inference threads
#   .\run.ps1 -Backend amd                      # NPU, AMD ONNX model, cache demo
#   .\run.ps1 -Backend onnx-static -Device cpu  # unchanged static ONNX via ONNX Runtime
#   .\run.ps1 -Model <model_dir> -Audio <wav> -NoCache -NoResults

[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Audio = "",
    [ValidateSet("auto", "intel", "intel-onnx", "onnx-static", "amd", "qualcomm")][string]$Backend = "intel",
    [ValidateSet("npu", "gpu", "cpu")][string]$Device = "npu",
    [int]$Runs = 5,
    [int]$Threads = 0,
    [string]$Configuration = "Release",
    [string]$Results = "",
    [string]$Label = "",
    [string]$Ref = "",
    [switch]$NoCache,
    [switch]$NoResults
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
if (-not $Model) {
    switch ($Backend) {
        "amd" { $Model = Join-Path $root "models\whisper-tiny-amd" }
        "intel-onnx" { $Model = Join-Path $root "models\whisper-tiny-en-onnx" }
        "onnx-static" { $Model = Join-Path $root "models\whisper-tiny-en-static-onnx" }
        default { $Model = Join-Path $root "models\whisper-tiny-en-ov" }
    }
}
if (-not $Audio) { $Audio = Join-Path $root "models\jfk.wav" }
if (-not $Results) { $Results = Join-Path $root "results\benchmark-results.csv" }

$exe = Join-Path $root "build\x64\$Configuration\WhisperNpuHal.App.exe"
if (-not (Test-Path $exe)) { Write-Host "Not built: $exe  (run .\scripts\build.ps1 or bootstrap.ps1)" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Model)) { Write-Host "Model not found: $Model  (run .\scripts\get-model.ps1)" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Audio)) { Write-Host "Audio not found: $Audio  (run .\scripts\get-audio.ps1)" -ForegroundColor Red; exit 1 }

$extraArgs = @()
if (-not $NoCache) {
    # Scoped by backend AND device: Intel/AMD (and NPU/GPU/CPU within each) compile
    # to incompatible blobs, so a shared cache dir would silently cross-contaminate.
    $cacheDir = Join-Path $root "build\cache\$Backend\$Device"
    $extraArgs += @("--cache", $cacheDir)
}
if ($Threads -gt 0) { $extraArgs += @("--threads", "$Threads") }
if (-not $NoResults) {
    $extraArgs += @("--results", $Results)
    if ($Label) { $extraArgs += @("--label", $Label) }
}
if ($Ref) { $extraArgs += @("--ref", $Ref) }

& $exe $Model $Audio $Backend $Device $Runs @extraArgs
