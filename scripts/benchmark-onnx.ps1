# The default cross-platform benchmark: runs the three portable ONNX models this
# project ships -- Whisper tiny.en (ASR), the Text Scam Classifier (TSC), and the
# FakeAudio / Generated Audio Detector -- through the SAME C++ app on the SAME
# ONNX Runtime, so one command produces a comparable picture on any box (x64 or
# arm64, CPU / GPU / NPU).
#
# Whisper goes through the app's transcribe path (onnx-static backend); the two
# classifiers go through --classify, replaying the pre-baked, validated fixture
# tensors from scripts/experiments/dump_fixtures.py. Fixtures are auto-generated
# on first run (needs the .venv + the deepfake models present).
#
#   .\scripts\benchmark-onnx.ps1                       # cpu, all three models
#   .\scripts\benchmark-onnx.ps1 -Device gpu           # DirectML / OpenVINO GPU
#   .\scripts\benchmark-onnx.ps1 -Device npu,cpu       # sweep several devices
#   .\scripts\benchmark-onnx.ps1 -Only whisper         # just the ASR model
#   .\scripts\benchmark-onnx.ps1 -RegenerateFixtures   # rebuild classifier fixtures first

[CmdletBinding()]
param(
    [ValidateSet("npu", "gpu", "cpu")][string[]]$Device = @("cpu"),
    [ValidateSet("whisper", "tsc", "fakeaudio", "all")][string[]]$Only = @("all"),
    [int]$Runs = 5,
    [int]$ClassifierRuns = 20,
    [string]$Configuration = "Release",
    [string]$Provider = "",
    [string]$Audio = "",
    [string]$Results = "",
    [string]$ClassifierResults = "",
    [switch]$RegenerateFixtures,
    [switch]$NoResults
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "benchmark.lib.ps1")
Initialize-BenchmarkConsole

$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root "build\x64\$Configuration\WhisperNpuHal.App.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Not built: $exe  (run .\scripts\build.ps1 -EnableOrt or bootstrap.ps1)" -ForegroundColor Red
    exit 1
}

if ($Only -contains "all") { $Only = @("whisper", "tsc", "fakeaudio") }
if (-not $Results) { $Results = Join-Path $root "results\benchmark-results.csv" }
if (-not $ClassifierResults) { $ClassifierResults = Join-Path $root "results\deepfake-benchmark-cpp.csv" }
if (-not $Audio) { $Audio = Join-Path $root "models\eval\ls_000.wav" }

# Whisper ONNX comes in two flavors of the SAME model, both benchmarked here:
#   static  : no-KV recompute (fixed shapes) -> NPU-compilable, slower.
#   dynamic : with-past KV cache (growing shapes) -> faster on CPU/GPU, but the
#             NPU compilers reject dynamic shapes, so it is a CPU/GPU-only citizen.
$whisperStatic = Join-Path $root "models\whisper\en-static-onnx"
$whisperDynamic = Join-Path $root "models\whisper\en-onnx"
$whisperVariants = @(
    @{ Backend = "onnx-static";  Model = $whisperStatic;  Devices = @("npu", "gpu", "cpu"); Label = "onnx-static" }
    @{ Backend = "onnx-dynamic"; Model = $whisperDynamic; Devices = @("gpu", "cpu");        Label = "onnx-dynamic" }
)
$fixRoot = Join-Path $root "models\deepfake\fixtures"
$classifierFix = @{ tsc = (Join-Path $fixRoot "tsc"); fakeaudio = (Join-Path $fixRoot "fakeaudio") }

# --- best-effort whisper reference (WER) from the eval manifest --------------
function Get-EvalRef {
    param([string]$AudioPath)
    $evalJsonl = Join-Path $root "models\eval\eval.jsonl"
    if (-not (Test-Path $evalJsonl)) { return "" }
    $id = [System.IO.Path]::GetFileNameWithoutExtension($AudioPath)
    foreach ($line in Get-Content $evalJsonl) {
        if (-not $line.Trim()) { continue }
        try { $row = $line | ConvertFrom-Json } catch { continue }
        if ($row.id -eq $id) { return [string]$row.ref }
    }
    return ""
}

# --- ensure classifier fixtures exist (generate via the validated Python) ----
function Ensure-Fixtures {
    param([string[]]$Models)
    $need = @()
    foreach ($m in $Models) {
        if ($RegenerateFixtures -or -not (Test-Path (Join-Path $classifierFix[$m] "model.tsv"))) { $need += $m }
    }
    if ($need.Count -eq 0) { return }

    $vpy = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path $vpy)) {
        Write-Host "Fixtures missing for: $($need -join ', '); no .venv to generate them (run bootstrap.ps1)." -ForegroundColor Yellow
        return
    }
    Write-Host "Generating classifier fixtures: $($need -join ', ') ..." -ForegroundColor Cyan
    $env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
    & $vpy (Join-Path $root "scripts\experiments\dump_fixtures.py") --models @need
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Fixture generation failed (models present? deepfake pipeline deps installed?)." -ForegroundColor Yellow
    }
}

$ran = @()
$skipped = @()

$wantClassifiers = @($Only | Where-Object { $_ -in @("tsc", "fakeaudio") })
if ($wantClassifiers.Count -gt 0) { Ensure-Fixtures -Models $wantClassifiers }

foreach ($dev in $Device) {
    Write-Host "`n==================== device: $dev ====================" -ForegroundColor Magenta

    if ($Only -contains "whisper") {
        $ref = Get-EvalRef -AudioPath $Audio
        foreach ($v in $whisperVariants) {
            $tag = "whisper-$($v.Label)/$dev"
            if ($v.Devices -notcontains $dev) {
                # dynamic on NPU: the KV-cache dynamic shapes are rejected by the NPU
                # compiler -- expected, so we skip rather than record a demoted CPU run.
                Write-Host "`n--- Whisper $($v.Label): skipped on $dev (dynamic KV shapes are not NPU-compilable) ---" -ForegroundColor DarkYellow
                $skipped += $tag
                continue
            }
            if (-not (Test-Path $v.Model)) {
                Write-Host "Whisper model missing: $($v.Model)  (run .\scripts\get-models.ps1 / export-onnx-dynamic.ps1)" -ForegroundColor Yellow
                $skipped += $tag
                continue
            }
            Write-Host "`n--- Whisper tiny.en ($($v.Label), ASR) ---" -ForegroundColor Cyan
            $runArgs = @{
                Backend       = $v.Backend
                Device        = $dev
                Runs          = $Runs
                Model         = $v.Model
                Audio         = $Audio
                Configuration = $Configuration
            }
            if ($Provider) { $runArgs.Provider = $Provider }
            if ($NoResults) { $runArgs.NoResults = $true }
            else { $runArgs.Results = $Results; $runArgs.Label = $v.Label }
            if ($ref) { $runArgs.Ref = $ref }
            & (Join-Path $PSScriptRoot "run.ps1") @runArgs
            $ran += $tag
        }
    }

    foreach ($m in @("tsc", "fakeaudio")) {
        if ($Only -notcontains $m) { continue }
        $fix = $classifierFix[$m]
        if (-not (Test-Path (Join-Path $fix "model.tsv"))) {
            Write-Host "$m fixture missing: $fix  (needs .venv + models/deepfake/$m)" -ForegroundColor Yellow
            $skipped += "$m/$dev"
            continue
        }
        Write-Host "`n--- $m (classifier) ---" -ForegroundColor Cyan
        $clsArgs = @("--classify", $fix, $dev, "$ClassifierRuns")
        if ($Provider) { $clsArgs += @("--provider", $Provider) }
        if (-not $NoResults) { $clsArgs += @("--results", $ClassifierResults) }
        & $exe @clsArgs
        if ($LASTEXITCODE -eq 0) { $ran += "$m/$dev" } else { $skipped += "$m/$dev" }
    }
}

Write-Host "`n==================== summary ====================" -ForegroundColor Magenta
Write-Host "ran    : $(if ($ran.Count) { $ran -join ', ' } else { '(nothing)' })" -ForegroundColor Green
if ($skipped.Count) { Write-Host "skipped: $($skipped -join ', ')" -ForegroundColor Yellow }
if (-not $NoResults) {
    Write-Host "ASR ledger        : $Results" -ForegroundColor DarkGray
    Write-Host "classifier ledger : $ClassifierResults" -ForegroundColor DarkGray
}
