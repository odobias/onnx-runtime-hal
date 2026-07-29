# Bake CPU fp32 Whisper-static hypotheses + WER/CER into src/workloads/eval/eval.jsonl
# so later suite runs can compare against a frozen baseline.
#
# Without -BaselineKey the flat baseline_* fields are written (the tiny.en
# static-onnx reference shared by every workload that ships those weights).
# With -BaselineKey the hypotheses land under baselines.<key> instead, so a
# package with different weights gets its own frozen reference and is never
# graded against another package's transcripts. The manifest workload must
# declare the matching "baselineKey" for the suite to use it.
#
#   .\tools\eval\bake-whisper-baselines.ps1
#   .\tools\eval\bake-whisper-baselines.ps1 -Device cpu -Runtime bundled
#   .\tools\eval\bake-whisper-baselines.ps1 `
#       -ModelDir artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s `
#       -BaselineKey static-onnx-tiny-multi-7s

[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu", "npu")][string]$Device = "cpu",
    [ValidateSet("bundled", "winml")][string]$Runtime = "bundled",
    [string]$ModelDir = "",
    [string]$BaselineKey = ""
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

if ($BaselineKey -and -not $ModelDir) {
    throw "-BaselineKey requires -ModelDir so the key cannot be baked from the wrong package"
}
$modelRel = if ($ModelDir) {
    ($ModelDir -replace '\\', '/').TrimEnd('/')
} else {
    "artifacts/workloads/whisper/models/static-onnx"
}
$modelDir = if ([System.IO.Path]::IsPathRooted($modelRel)) {
    $modelRel
} else {
    Join-Path $root ($modelRel -replace '/', '\')
}
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
        $clipPath = Join-Path $root "artifacts\workloads\speech\$($eval.id).wav"
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

# Clips go through a UTF-8 JSONL file, not --eval-clip: a reference containing a
# double quote is truncated by the Windows command line, which would silently bake a
# baseline against a partial reference.
$setPath = Join-Path $cache "clips.jsonl"
$clipSpecs | ForEach-Object {
    [pscustomobject]@{ id = $_.id; audio = $_.audio; ref = $_.ref } |
        ConvertTo-Json -Compress -Depth 3
} | Set-Content -LiteralPath $setPath -Encoding UTF8

$args = @(
    "run", "whisper", $modelDir, $clipSpecs[0].audio,
    "onnx-static", $Device, "1",
    "--cache", $cache, "--json", "--json-output", $jsonOutput,
    "--eval-set", $setPath
)

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
    model = $modelRel
    baked_at_utc = [DateTime]::UtcNow.ToString("o")
    contract = "whisper-eval-v1"
}

$updated = foreach ($eval in $evalRows) {
    $id = [string]$eval.id
    $clip = $byId[$id]
    if (-not $clip) { throw "Bake result missing clip '$id'" }
    $audioRel = "artifacts/workloads/speech/$id.wav"
    if (-not (Test-Path -LiteralPath (Join-Path $root ($audioRel -replace '/', '\')))) {
        $audioRel = [string]$eval.audio
    }
    $hyp = if ($clip.text) { [string]$clip.text } else { [string]$clip.transcription }
    # Multilingual packages detect a language per clip; freeze it so a later run
    # that hears a different one is flagged instead of silently graded on WER.
    # English-only packages report none and stay uncompared.
    $lang = [string]$clip.language
    $task = [string]$clip.task
    if (-not $BaselineKey) {
        $flat = [ordered]@{
            id = $id
            audio = $audioRel
            ref = [string]$eval.ref
            duration_s = [double]$eval.duration_s
            baseline_hyp = $hyp
            baseline_wer = [double]$clip.wer
            baseline_cer = [double]$clip.cer
        }
        if ($lang) { $flat["baseline_lang"] = $lang }
        if ($task) { $flat["baseline_task"] = $task }
        $flat["baseline_source"] = $source
        $flat
        continue
    }
    # Keep every existing field -- other workloads' keys and the flat legacy
    # mirror included -- and replace only this key's entry.
    $row = [ordered]@{}
    foreach ($property in $eval.PSObject.Properties) {
        if ($property.Name -eq "baselines") { continue }
        $row[$property.Name] = $property.Value
    }
    $row["audio"] = $audioRel
    $map = [ordered]@{}
    if (($eval.PSObject.Properties.Name -contains "baselines") -and $eval.baselines) {
        foreach ($property in $eval.baselines.PSObject.Properties) {
            $map[$property.Name] = $property.Value
        }
    }
    $entry = [ordered]@{
        hyp = $hyp
        wer = [double]$clip.wer
        cer = [double]$clip.cer
    }
    if ($lang) { $entry["lang"] = $lang }
    if ($task) { $entry["task"] = $task }
    $entry["source"] = $source
    $map[$BaselineKey] = $entry
    $row["baselines"] = $map
    $row
}

$backup = Join-Path $root ("src\workloads\eval\eval.jsonl.bak-{0:yyyyMMdd-HHmmss}" -f (Get-Date))
Copy-Item -LiteralPath $evalPath -Destination $backup -Force
$updated | ForEach-Object {
    ($_ | ConvertTo-Json -Compress -Depth 10)
} | Set-Content -LiteralPath $evalPath -Encoding UTF8

$agg = Measure-BenchmarkClips $native.payload.clips
Write-Host "Wrote baselines for $($updated.Count) clips -> $evalPath" -ForegroundColor Green
Write-Host ("Corpus baseline WER={0:P2} CER={1:P2} (backup {2})" -f `
    $agg.wer, $agg.cer, $backup) -ForegroundColor Cyan
