# Runs the built whisper_hal app. Runtime DLLs are copied next to the exe by the
# build, so no environment setup is required. Model/audio default to the repo-local
# copies produced by get-model.ps1 / get-audio.ps1.
#
#   .\run.ps1                                   # NPU, Intel exported model, cache demo
#   .\run.ps1 -Device cpu -Backend intel
#   .\run.ps1 -Device cpu -Threads 8            # pin OpenVINO CPU inference threads
#   .\run.ps1 -Backend amd                      # NPU, AMD ONNX model, cache demo
#   .\run.ps1 -Backend onnx-static -Device cpu  # unchanged static ONNX via ONNX Runtime
#   .\run.ps1 -Backend onnx-dynamic -Device gpu # with-past KV-cache ONNX (CPU/GPU only; NPU rejects it)
#   .\run.ps1 -Backend onnx-static -Device npu -Provider VitisAIExecutionProvider
#   .\run.ps1 -Backend qualcomm -Device npu   # static ONNX via Plugin QNN EP (Snapdragon)
#   .\run.ps1 -Model <model_dir> -Audio <wav> -NoCache -NoResults

[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Audio = "",
    [ValidateSet("auto", "intel", "intel-onnx", "onnx-static", "onnx-dynamic", "amd", "qualcomm")][string]$Backend = "intel",
    [ValidateSet("npu", "gpu", "cpu")][string]$Device = "npu",
    [int]$Runs = 5,
    [int]$Threads = 0,
    [string]$Provider = "",
    [ValidateSet("x64", "x64-ovep", "ARM64", "ARM64-ovep")][string]$Platform = "x64",
    [string]$Configuration = "Release",
    [string]$Results = "",
    [string]$Label = "",
    [string]$Ref = "",
    [switch]$NoCache,
    [switch]$NoResults
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "benchmark.lib.ps1")
Initialize-BenchmarkConsole

$root = Split-Path $PSScriptRoot -Parent
if (-not $Model) {
    switch ($Backend) {
        "amd" { $Model = Join-Path $root "models\whisper\amd" }
        "intel-onnx" { $Model = Join-Path $root "models\whisper\en-onnx" }
        "onnx-static" { $Model = Join-Path $root "models\whisper\en-static-onnx" }
        "onnx-dynamic" { $Model = Join-Path $root "models\whisper\en-onnx" }
        "qualcomm" { $Model = Join-Path $root "models\whisper\en-static-onnx" }
        default { $Model = Join-Path $root "models\whisper\en-ov" }
    }
}
if (-not $Audio) { $Audio = Join-Path $root "models\audio\jfk.wav" }
if (-not $Results) { $Results = Join-Path $root "results\benchmark-results.csv" }

$exe = Join-Path $root "build\$Platform\$Configuration\WhisperNpuHal.App.exe"
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
if ($Provider) { $extraArgs += @("--provider", $Provider) }
if (-not $NoResults) {
    $extraArgs += @("--results", $Results)
    if ($Label) { $extraArgs += @("--label", $Label) }
}
if ($Ref) { $extraArgs += @("--ref", $Ref) }

# The ORT OpenVINO EP prints benign warnings to stderr (e.g. "some shape nodes were
# assigned to CPU"). Under $ErrorActionPreference='Stop', a stderr line from a native
# command surfaces as a terminating NativeCommandError and would abort before we ever
# see the transcription/timing. The process exit code is the real success signal, so
# relax error handling across the native call and gate on $LASTEXITCODE instead.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $exe $Model $Audio $Backend $Device $Runs @extraArgs
$appExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($appExit -ne 0) { Write-Host "App exited with code $appExit ($exe)" -ForegroundColor Yellow; exit $appExit }
