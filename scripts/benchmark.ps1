# Benchmark harness: runs every (variant x device x clip) from the manifest against
# the labeled eval set and produces CSV + Markdown reports comparing performance,
# self-confidence, and accuracy (WER/CER) per quantization method.
#
# Backend-agnostic by construction: it reads `backend` from each manifest entry and
# passes it straight to the runner, so AMD/Qualcomm variants drop in with no changes
# here. Unsupported (variant x device) combos are recorded, not fatal.
#
#   .\benchmark.ps1                       # all variants, their listed devices
#   .\benchmark.ps1 -Devices NPU          # restrict devices
#   .\benchmark.ps1 -Runs 5 -MaxClips 10

[CmdletBinding()]
param(
    [string]$Manifest = "",
    [string]$EvalSet = "",
    [string[]]$Devices = @(),      # empty = use each variant's own device list
    [int]$Runs = 3,
    [int]$MaxClips = 0,            # 0 = all clips in the eval set
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
if (-not $Manifest) { $Manifest = Join-Path $root "models\manifest.json" }
if (-not $EvalSet) { $EvalSet = Join-Path $root "models\eval\eval.jsonl" }
$exe = Join-Path $root "build\x64\$Configuration\WhisperNpuHal.App.exe"

foreach ($p in @($Manifest, $EvalSet, $exe)) {
    if (-not (Test-Path $p)) { Write-Host "Missing: $p" -ForegroundColor Red; exit 1 }
}

$manifestObj = Get-Content $Manifest -Raw | ConvertFrom-Json
$clips = Get-Content $EvalSet | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
if ($MaxClips -gt 0 -and $clips.Count -gt $MaxClips) { $clips = $clips[0..($MaxClips - 1)] }
Write-Host ("Manifest: {0} variant(s) | Eval: {1} clip(s) | Runs: {2}" -f `
        $manifestObj.variants.Count, $clips.Count, $Runs) -ForegroundColor Cyan

$reportsDir = Join-Path $root "results"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
$sharedCsv = Join-Path $reportsDir "benchmark-results.csv"
$cacheRoot = Join-Path $root "cache"

function Invoke-Clip($modelDir, $audio, $backend, $device, $runs, $cacheDir, $ref, $resultsCsv, $label) {
    $a = @($modelDir, $audio, $backend, $device, "$runs", "--cache", $cacheDir, "--ref", $ref, "--json")
    if ($resultsCsv) { $a += @("--results", $resultsCsv, "--label", $label) }
    $raw = & $exe @a 2>$null
    $line = ($raw | Where-Object { $_ -match '^\{"ok"' } | Select-Object -Last 1)
    if (-not $line) { return [pscustomobject]@{ ok = $false; error = "no-json-output" } }
    return $line | ConvertFrom-Json
}

$summary = @()
$detail = @()

foreach ($v in $manifestObj.variants) {
    $modelDir = Join-Path $root ($v.model_dir -replace '/', '\')
    if (-not (Test-Path $modelDir)) {
        Write-Host "  ! $($v.id): model_dir missing ($modelDir); skipping" -ForegroundColor Yellow
        continue
    }
    $devList = if ($Devices.Count) { $Devices } else { $v.devices }

    foreach ($dev in $devList) {
        $tag = "$($v.id)-$dev"
        Write-Host "== $tag ==" -ForegroundColor White
        $cacheDir = Join-Path $cacheRoot $tag
        if (Test-Path $cacheDir) { Remove-Item $cacheDir -Recurse -Force }  # force a true cold compile

        $rows = @()
        $status = "ok"
        $errMsg = ""
        $first = $true
        $coldLoad = $null; $warmLoad = $null

        foreach ($c in $clips) {
            $audio = Join-Path $root ($c.audio -replace '/', '\')
            if (-not (Test-Path $audio)) { continue }
            # Emit one canonical row per (variant x device) into the shared CSV using
            # the first clip; the rest feed only the aggregate report below.
            $emitCsv = if ($first) { $sharedCsv } else { $null }
            $r = Invoke-Clip $modelDir $audio $v.backend $dev $Runs $cacheDir $c.ref $emitCsv $v.id
            if (-not $r.ok) {
                $status = "unsupported/error"; $errMsg = $r.error
                Write-Host ("   {0}: {1}" -f $c.id, $r.error) -ForegroundColor Yellow
                break
            }
            if ($first) { $coldLoad = $r.load_cold_s; $warmLoad = $r.load_warm_s; $first = $false }
            $rows += $r
            $detail += [pscustomobject]@{
                variant = $v.id; precision = $v.precision; backend = $v.backend; device = $dev
                clip = $c.id; mean_ms = [math]::Round($r.mean_ms, 1); rtf = [math]::Round($r.rtf, 4)
                avg_logprob = [math]::Round($r.avg_logprob, 4); wer = [math]::Round($r.wer * 100, 2)
                cer = [math]::Round($r.cer * 100, 2)
            }
        }

        if ($rows.Count -eq 0) {
            $summary += [pscustomobject]@{
                variant = $v.id; precision = $v.precision; backend = $v.backend; device = $dev
                status = $status; clips = 0; size_mb = $v.size_mb
                cold_s = $null; warm_s = $null; mean_ms = $null; rtf = $null; xrt = $null
                tps = $null; avg_logprob = $null; wer_pct = $null; cer_pct = $null; error = $errMsg
            }
            continue
        }

        $meanMs = ($rows | Measure-Object mean_ms -Average).Average
        $meanRtf = ($rows | Measure-Object rtf -Average).Average
        $meanTps = ($rows | Measure-Object throughput_tps -Average).Average
        $meanLp = ($rows | Measure-Object avg_logprob -Average).Average
        $wEdits = ($rows | Measure-Object word_edits -Sum).Sum
        $wRef = ($rows | Measure-Object ref_words -Sum).Sum
        $cEdits = ($rows | Measure-Object char_edits -Sum).Sum
        $cRef = ($rows | Measure-Object ref_chars -Sum).Sum

        $summary += [pscustomobject]@{
            variant = $v.id; precision = $v.precision; backend = $v.backend; device = $dev
            status = "ok"; clips = $rows.Count; size_mb = $v.size_mb
            cold_s = [math]::Round($coldLoad, 2); warm_s = [math]::Round($warmLoad, 2)
            mean_ms = [math]::Round($meanMs, 1); rtf = [math]::Round($meanRtf, 4)
            xrt = [math]::Round((1.0 / [math]::Max($meanRtf, 1e-9)), 1); tps = [math]::Round($meanTps, 1)
            avg_logprob = [math]::Round($meanLp, 4)
            wer_pct = if ($wRef) { [math]::Round(100.0 * $wEdits / $wRef, 2) } else { $null }
            cer_pct = if ($cRef) { [math]::Round(100.0 * $cEdits / $cRef, 2) } else { $null }
            error = ""
        }
        Write-Host ("   ok: {0} clips | {1} ms | WER {2}% | conf {3}" -f `
                $rows.Count, [math]::Round($meanMs, 1),
            $(if ($wRef) { [math]::Round(100.0 * $wEdits / $wRef, 2) } else { "n/a" }),
            [math]::Round($meanLp, 4)) -ForegroundColor Green
    }
}

# --- Write reports ---
$csvPath = Join-Path $reportsDir "quantization-benchmark.csv"
$detailCsv = Join-Path $reportsDir "quantization-benchmark-detail.csv"
$mdPath = Join-Path $reportsDir "quantization-benchmark.md"
$summary | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
$detail | Export-Csv -Path $detailCsv -NoTypeInformation -Encoding UTF8

$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("# Whisper NPU HAL - quantization benchmark")
[void]$md.AppendLine("")
[void]$md.AppendLine("- Model: ``$($manifestObj.model)``")
[void]$md.AppendLine("- Eval clips: $($clips.Count) | Runs/clip: $Runs | Generated: $(Get-Date -Format s)")
[void]$md.AppendLine("- Confidence = mean per-token log-prob (self-reported; higher = more confident, not calibrated truth).")
[void]$md.AppendLine("- WER/CER micro-averaged over clips after normalization.")
[void]$md.AppendLine("")
[void]$md.AppendLine("| Variant | Prec | Backend | Device | Status | Size MB | Cold s | Warm s | Mean ms | xRT | tok/s | Conf | WER % | CER % |")
[void]$md.AppendLine("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
foreach ($s in $summary) {
    $f = { param($x) if ($null -eq $x) { "-" } else { $x } }
    [void]$md.AppendLine(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} | {13} |" -f `
                $s.variant, $s.precision, $s.backend, $s.device, $s.status, (& $f $s.size_mb),
            (& $f $s.cold_s), (& $f $s.warm_s), (& $f $s.mean_ms), (& $f $s.xrt), (& $f $s.tps),
            (& $f $s.avg_logprob), (& $f $s.wer_pct), (& $f $s.cer_pct)))
}
[void]$md.AppendLine("")
[void]$md.AppendLine("_Caching: cold = first compile, warm = cache hit. See ``cache/`` and OpenVINO ``ov::cache_dir``._")
Set-Content -Path $mdPath -Value $md.ToString() -Encoding UTF8

Write-Host "`nReports written:" -ForegroundColor Green
Write-Host "  $csvPath"
Write-Host "  $detailCsv"
Write-Host "  $mdPath"
Write-Host ""
$summary | Format-Table variant, precision, device, status, size_mb, cold_s, warm_s, mean_ms, xrt, avg_logprob, wer_pct -AutoSize
