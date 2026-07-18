# Bake CPU fp32 Whisper-static hypotheses + WER/CER into src/workloads/eval/eval.jsonl
# so later suite runs can compare against a frozen baseline.
#
#   .\tools\eval\bake-whisper-baselines.ps1
#   .\tools\eval\bake-whisper-baselines.ps1 -Device cpu -Runtime bundled

[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu", "npu")][string]$Device = "cpu",
    [ValidateSet("bundled", "winml")][string]$Runtime = "bundled"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. (Join-Path $root "benchmark\lib\harness.ps1")
Initialize-BenchmarkConsole | Out-Null

$evalPath = Join-Path $root "src\workloads\eval\eval.jsonl"
if (-not (Test-Path -LiteralPath $evalPath)) {
    throw "Eval set missing: $evalPath (run tools/fetch/get-eval-set.ps1 first)"
}

$hostVendor = [string](Get-BenchmarkHardware).cpu.vendor
$hostArch = Get-BenchmarkHostArchitecture
$platform = switch ($Runtime) {
    "winml" { "$hostArch-winml" }
    default {
        switch ($hostVendor) {
            "AMD" { "$hostArch-amd" }
            "Qualcomm" { "$hostArch-qualcomm" }
            default { $hostArch }
        }
    }
}
$exe = Join-Path $root "artifacts\build\$platform\Release\NpuInferenceBench.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    throw "Benchmark executable missing: $exe"
}

$modelDir = Join-Path $root "artifacts\workloads\whisper\models\static-onnx"
if (-not (Test-Path -LiteralPath (Join-Path $modelDir "encoder_model.onnx"))) {
    throw "Whisper static model missing: $modelDir"
}

$evalRows = @(Get-Content -LiteralPath $evalPath -Encoding UTF8 |
    Where-Object { $_.Trim() } |
    ForEach-Object { $_ | ConvertFrom-Json })
if (-not $evalRows.Count) { throw "eval.jsonl is empty" }

$clipSpecs = foreach ($eval in $evalRows) {
    $clipPath = Join-Path $root (([string]$eval.audio -replace '/', '\'))
    if (-not (Test-Path -LiteralPath $clipPath)) {
        $clipPath = Join-Path $root "artifacts\workloads\eval\$($eval.id).wav"
    }
    if (-not (Test-Path -LiteralPath $clipPath)) {
        throw "Eval audio missing for $($eval.id)"
    }
    [pscustomobject]@{
        id = [string]$eval.id
        audio = $clipPath
        ref = [string]$eval.ref
    }
}

$cache = Join-Path $root "artifacts\build\cache\baseline-bake\$platform\$Device"
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$jsonOutput = Join-Path $cache "batch-result.json"
Remove-Item -LiteralPath $jsonOutput -Force -ErrorAction SilentlyContinue

$args = @(
    "run", "whisper", $modelDir, $clipSpecs[0].audio,
    "onnx-static", $Device, "1",
    "--cache", $cache, "--json", "--json-output", $jsonOutput
)
foreach ($clip in $clipSpecs) {
    $args += @("--eval-clip", $clip.id, $clip.audio, $clip.ref)
}

Write-Host "Baking baselines via $exe ($Runtime/$Device)..." -ForegroundColor Cyan
$native = Invoke-BenchmarkNativeJson -Exe $exe -Arguments $args `
    -JsonOutputPath $jsonOutput -EchoOutput
if (-not $native.succeeded -or -not $native.payload.clips) {
    $err = if ($native.payload.error) { $native.payload.error } else { "exit $($native.exit_code)" }
    throw "Baseline bake failed: $err"
}

$byId = @{}
foreach ($clip in @($native.payload.clips)) {
    $clipId = if ($clip.eval_id) { [string]$clip.eval_id } else { [string]$clip.id }
    if (-not $clipId) { throw "Bake clip is missing eval_id/id" }
    $byId[$clipId] = $clip
}

$source = [ordered]@{
    runtime = $Runtime
    device = $Device
    resolved_provider = $(
        if ($native.payload.clips[0].resolved_provider) {
            [string]$native.payload.clips[0].resolved_provider
        } elseif ($native.payload.clips[0].execution_provider) {
            [string]$native.payload.clips[0].execution_provider
        } elseif ($native.payload.clips[0].device) {
            [string]$native.payload.clips[0].device
        } else { "" }
    )
    model = "artifacts/workloads/whisper/models/static-onnx"
    baked_at_utc = [DateTime]::UtcNow.ToString("o")
    contract = "whisper-eval-v1"
}

$updated = foreach ($eval in $evalRows) {
    $id = [string]$eval.id
    $clip = $byId[$id]
    if (-not $clip) { throw "Bake result missing clip '$id'" }
    $audioRel = "artifacts/workloads/eval/$id.wav"
    if (-not (Test-Path -LiteralPath (Join-Path $root ($audioRel -replace '/', '\')))) {
        $audioRel = [string]$eval.audio
    }
    $hyp = if ($clip.text) { [string]$clip.text } else { [string]$clip.transcription }
    [ordered]@{
        id = $id
        audio = $audioRel
        ref = [string]$eval.ref
        duration_s = [double]$eval.duration_s
        baseline_hyp = $hyp
        baseline_wer = [double]$clip.wer
        baseline_cer = [double]$clip.cer
        baseline_source = $source
    }
}

$backup = Join-Path $root ("src\workloads\eval\eval.jsonl.bak-{0:yyyyMMdd-HHmmss}" -f (Get-Date))
Copy-Item -LiteralPath $evalPath -Destination $backup -Force
$updated | ForEach-Object {
    ($_ | ConvertTo-Json -Compress -Depth 6)
} | Set-Content -LiteralPath $evalPath -Encoding UTF8

$agg = Measure-BenchmarkClips $native.payload.clips
Write-Host "Wrote baselines for $($updated.Count) clips -> $evalPath" -ForegroundColor Green
Write-Host ("Corpus baseline WER={0:P2} CER={1:P2} (backup {2})" -f `
    $agg.wer, $agg.cer, $backup) -ForegroundColor Cyan
