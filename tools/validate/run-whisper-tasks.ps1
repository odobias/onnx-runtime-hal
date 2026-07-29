# Exercise the shipping runner on non-English speech for both Whisper tasks:
#
#   transcribe -> verbatim in the spoken language, scored against the FLEURS
#                 reference (this is what the product does)
#   translate  -> English out of foreign speech, scored against the parallel
#                 FLoRes English sentence (ref_en in the manifest)
#
# Language is always auto-detected, never pinned, so detection is under test too.
#
#   .\tools\validate\run-whisper-tasks.ps1
#   .\tools\validate\run-whisper-tasks.ps1 -Device npu
#   .\tools\validate\run-whisper-tasks.ps1 -Task transcribe
#   .\tools\validate\run-whisper-tasks.ps1 -Window full   # one long pass, for comparison

[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu", "npu")][string]$Device = "cpu",
    [ValidateSet("bundled", "winml")][string]$Runtime = "bundled",
    [ValidateSet("transcribe", "translate", "both")][string]$Task = "both",
    # product = the shipping path: 7s windows with 1s overlap, stitched.
    # full = decode the whole clip in one pass; only useful to price windowing.
    [ValidateSet("product", "full")][string]$Window = "product",
    [string]$ModelDir = "artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s",
    [string]$Manifest = "artifacts/workloads/speech/multilingual/manifest.jsonl"
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

$manifestPath = Join-Path $root ($Manifest -replace '/', '\')
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Multilingual manifest missing: $manifestPath (run tools/fetch/fetch_multilingual_speech.py)"
}
$modelDir = Join-Path $root ($ModelDir -replace '/', '\')
if (-not (Test-Path -LiteralPath (Join-Path $modelDir "encoder_model.onnx"))) {
    throw "Whisper static model missing: $modelDir"
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
if (-not (Test-Path -LiteralPath $exe)) { throw "Benchmark executable missing: $exe" }

$clips = @(Get-Content -LiteralPath $manifestPath -Encoding UTF8 |
    Where-Object { $_.Trim() } |
    ForEach-Object { $_ | ConvertFrom-Json })
if (-not $clips.Count) { throw "Manifest is empty: $manifestPath" }

$tasks = if ($Task -eq "both") { @("transcribe", "translate") } else { @($Task) }

function Invoke-TaskRun {
    param(
        [Parameter(Mandatory)][string]$TaskName,
        [Parameter(Mandatory)][object[]]$Clips
    )
    # The engine refuses translate under the product window, since stitched fragments
    # are not a translation. Say so here rather than letting the runner fail.
    if ($TaskName -eq "translate" -and $Window -eq "product") {
        Write-Host ("`n=== task: translate -- skipped: not supported under the product window " +
            "(re-run with -Window full) ===") -ForegroundColor Yellow
        return $null
    }

    # translate is scored against the parallel English sentence, so a clip without
    # one is skipped rather than silently graded against its own language.
    $usable = @($Clips | Where-Object {
        $TaskName -ne "translate" -or [string]$_.ref_en
    })
    if (-not $usable.Count) {
        Write-Warning "$TaskName): no clip has a reference; skipping"
        return $null
    }

    $cache = Join-Path $root "artifacts\build\cache\whisper-tasks\$platform\$Device\$TaskName"
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    $jsonOutput = Join-Path $cache "batch-result.json"
    Remove-Item -LiteralPath $jsonOutput -Force -ErrorAction SilentlyContinue

    # References contain double quotes (FLoRes quotes speech), which argv truncates
    # silently -- hand the runner a UTF-8 JSONL file instead of --eval-clip triples.
    $setPath = Join-Path $cache "clips.jsonl"
    $lines = foreach ($clip in $usable) {
        $audio = Join-Path $root ([string]$clip.audio -replace '/', '\')
        if (-not (Test-Path -LiteralPath $audio)) { throw "Clip audio missing: $audio" }
        $ref = if ($TaskName -eq "translate") { [string]$clip.ref_en } else { [string]$clip.ref }
        [pscustomobject]@{ id = [string]$clip.id; audio = $audio; ref = $ref } |
            ConvertTo-Json -Compress -Depth 3
    }
    $lines | Set-Content -LiteralPath $setPath -Encoding UTF8

    $first = Join-Path $root ([string]$usable[0].audio -replace '/', '\')
    $argv = @(
        "run", "whisper", $modelDir, $first,
        "onnx-static", $Device, "1",
        "--cache", $cache, "--json", "--json-output", $jsonOutput,
        "--eval-set", $setPath
    )

    Write-Host "`n=== task: $TaskName ($Runtime/$Device, $Window window, language auto-detect) ===" -ForegroundColor Cyan
    $env:NPU_INFERENCE_BENCH_WHISPER_TASK = $TaskName
    $env:NPU_INFERENCE_BENCH_WHISPER_WINDOW = $Window
    try {
        $native = Invoke-BenchmarkNativeJson -Exe $exe -Arguments $argv -JsonOutputPath $jsonOutput
    } finally {
        Remove-Item Env:NPU_INFERENCE_BENCH_WHISPER_TASK -ErrorAction SilentlyContinue
        Remove-Item Env:NPU_INFERENCE_BENCH_WHISPER_WINDOW -ErrorAction SilentlyContinue
    }
    if (-not $native.succeeded -or -not $native.payload.clips) {
        $err = if ($native.payload.error) { $native.payload.error } else { "exit $($native.exit_code)" }
        throw "$TaskName run failed: $err"
    }

    $byId = @{}
    foreach ($clip in @($native.payload.clips)) {
        $id = if ($clip.eval_id) { [string]$clip.eval_id } else { [string]$clip.id }
        $byId[$id] = $clip
    }

    $rows = foreach ($clip in $usable) {
        $got = $byId[[string]$clip.id]
        if (-not $got) { throw "$TaskName run is missing clip '$($clip.id)'" }
        [pscustomobject]@{
            id = [string]$clip.id
            expected_lang = [string]$clip.lang
            detected_lang = [string]$got.language
            lang_ok = ([string]$got.language -eq [string]$clip.lang)
            reported_task = [string]$got.task
            windows = [int]$got.windows
            wer = [double]$got.wer
            cer = [double]$got.cer
            ref = if ($TaskName -eq "translate") { [string]$clip.ref_en } else { [string]$clip.ref }
            hyp = if ($got.text) { [string]$got.text } else { [string]$got.transcription }
            duration_s = [double]$clip.duration_s
        }
    }

    $agg = Measure-BenchmarkClips $native.payload.clips
    $langOk = @($rows | Where-Object { $_.lang_ok }).Count
    $taskOk = @($rows | Where-Object { $_.reported_task -eq $TaskName }).Count
    $totalWindows = ($rows | Measure-Object -Property windows -Sum).Sum
    Write-Host ("clips={0}  source-language detected={1}/{0}  task confirmed={2}/{0}  windows={3}  WER={4:P1} CER={5:P1}" -f `
        $rows.Count, $langOk, $taskOk, $totalWindows, $agg.wer, $agg.cer) -ForegroundColor DarkCyan
    foreach ($row in $rows) {
        $flag = if ($row.lang_ok) { " " } else { "!" }
        Write-Host ("{0} {1,-11} {2}->{3}  {4}w  WER={5,6:P1}  {6}" -f `
            $flag, $row.id, $row.expected_lang, $row.detected_lang, $row.windows, $row.wer, $row.hyp)
    }

    return [pscustomobject]@{
        task = $TaskName
        device = $Device
        runtime = $Runtime
        window_mode = $Window
        resolved_provider = [string]$native.payload.clips[0].resolved_provider
        clips = $rows.Count
        source_language_detected = $langOk
        task_confirmed = $taskOk
        windows = [int]$totalWindows
        wer = [double]$agg.wer
        cer = [double]$agg.cer
        rows = @($rows)
    }
}

$results = @()
foreach ($taskName in $tasks) {
    $result = Invoke-TaskRun -TaskName $taskName -Clips $clips
    if ($result) { $results += $result }
}
if (-not $results.Count) { throw "No task produced a result" }

$reportDir = Join-Path $root "results\reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$stem = "whisper-multi-7s-tasks-$Device-$Window"
$jsonPath = Join-Path $reportDir "$stem.json"
[pscustomobject]@{
    model = ($ModelDir -replace '\\', '/')
    manifest = ($Manifest -replace '\\', '/')
    device = $Device
    runtime = $Runtime
    window_mode = $Window
    language_mode = "auto-detect"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    tasks = @($results)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$md = [System.Collections.Generic.List[string]]::new()
$md.Add("# Whisper task check: static-onnx-tiny-multi-7s ($Device, $Window window)")
$md.Add("")
$md.Add("Runner: ``NpuInferenceBench.exe`` (the shipping static engine), language auto-detected per clip.")
$windowNote = if ($Window -eq "product") {
    "Audio is cut into 7s windows with 1s overlap and the transcripts stitched on their shared words, " +
    "so a clip costs one encoder pass per window."
} else {
    "The whole clip is decoded in a single pass -- not the product path, kept only to price windowing."
}
$md.Add($windowNote)
$md.Add("Source: FLEURS clips in ``$($Manifest -replace '\\', '/')``.")
$md.Add("")
foreach ($result in $results) {
    $scoredAgainst = if ($result.task -eq "translate") {
        "parallel FLoRes English sentence (``ref_en``)"
    } else {
        "FLEURS reference in the spoken language (``ref``)"
    }
    $md.Add("## task ``$($result.task)``")
    $md.Add("")
    $md.Add("Scored against the $scoredAgainst. Provider: ``$($result.resolved_provider)``.")
    $md.Add("")
    $md.Add(("Source language detected: **{0}/{1}** · windows decoded: **{2}** · mean WER: **{3:P1}** · mean CER: **{4:P1}**" -f `
        $result.source_language_detected, $result.clips, $result.windows, $result.wer, $result.cer))
    $md.Add("")
    $md.Add("| Clip | Spoken | Detected | Windows | WER | Output |")
    $md.Add("|------|--------|----------|---------|-----|--------|")
    foreach ($row in $result.rows) {
        $detected = if ($row.lang_ok) { $row.detected_lang } else { "**$($row.detected_lang)**" }
        $hyp = ($row.hyp -replace '\|', '\|')
        $md.Add(("| {0} | {1} | {2} | {3} | {4:P1} | {5} |" -f `
            $row.id, $row.expected_lang, $detected, $row.windows, $row.wer, $hyp))
    }
    $md.Add("")
}
$mdPath = Join-Path $reportDir "$stem.md"
$md | Set-Content -LiteralPath $mdPath -Encoding UTF8

Write-Host "`nWrote $mdPath" -ForegroundColor Green
Write-Host "Wrote $jsonPath" -ForegroundColor Green
