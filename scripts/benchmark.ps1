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
    [string]$Results = "",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

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
if (-not $Results) { $Results = Join-Path $reportsDir "benchmark-results.csv" }
$cacheRoot = Join-Path $root "cache"

function ConvertTo-CsvCell($Value) {
    $s = if ($null -eq $Value) { "" } else { [string]$Value }
    if ($s.IndexOfAny([char[]]",`"`r`n") -lt 0) { return $s }
    return '"' + ($s -replace '"', '""') + '"'
}

function Get-Optional($Object, [string]$Name, $Default = $null) {
    if ($null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
    return $Default
}

# Best-effort AC vs battery detection (battery => throttled clocks, so a run on
# battery can silently skew comparisons). "ac" / "battery" / "unknown".
function Get-PowerSource {
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        switch ([System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus) {
            'Online' { return 'ac' }
            'Offline' { return 'battery' }
            default { return 'unknown' }
        }
    } catch { return 'unknown' }
}

function Get-ModelPackage([string]$ModelDir) {
    if (-not $ModelDir) { return "" }
    return [System.IO.Path]::GetFileName(($ModelDir -replace '\\', '/').TrimEnd('/'))
}

function Get-InferredPrecision([string]$Package) {
    $p = $Package.ToLowerInvariant()
    if ($p -match 'int4') { return 'int4' }
    if ($p -match 'int8') { return 'int8' }
    if ($p -match 'fp16' -or $p -match '-f16') { return 'fp16' }
    if ($p -match 'static') { return 'fp32-static' }
    return 'fp32'
}

function Get-BenchmarkMeta($Variant, $ModelDir, $JsonRow) {
    $modelPackage = Get-ModelPackage $ModelDir
    $variantId = if ($Variant -and $Variant.id) { $Variant.id } else { "" }
    if (-not $variantId -and $JsonRow -and $JsonRow.variant_id) { $variantId = $JsonRow.variant_id }
    if (-not $variantId) { $variantId = $modelPackage }

    $baseModel = if ($manifestObj.model) { $manifestObj.model } else { "openai/whisper-tiny.en" }
    $precision = if ($Variant -and $Variant.precision) { $Variant.precision } else { Get-InferredPrecision $modelPackage }
    $quantMethod = if ($Variant -and $Variant.method) { $Variant.method } else { "FP32 baseline (inferred from package name)" }

    $executionProvider = ""
    $runtime = ""
    $modelFormat = ""
    $decodeStrategy = ""
    $maxContext = ""
    if ($JsonRow) {
        $executionProvider = Get-Optional $JsonRow "execution_provider" (Get-Optional $JsonRow "device" "")
        $runtime = Get-Optional $JsonRow "runtime" (Get-Optional $JsonRow "backend" "")
        $modelFormat = Get-Optional $JsonRow "model_format" ""
        $decodeStrategy = Get-Optional $JsonRow "decode_strategy" ""
        $maxContext = Get-Optional $JsonRow "max_context" ""
        if ($JsonRow.model_package) { $modelPackage = $JsonRow.model_package }
        if ($JsonRow.variant_id) { $variantId = $JsonRow.variant_id }
        if ($JsonRow.base_model) { $baseModel = $JsonRow.base_model }
        if ($JsonRow.precision) { $precision = $JsonRow.precision }
        if ($JsonRow.quant_method) { $quantMethod = $JsonRow.quant_method }
    }

    return [pscustomobject]@{
        model_package = $modelPackage
        variant_id = $variantId
        base_model = $baseModel
        precision = $precision
        quant_method = $quantMethod
        execution_provider = $executionProvider
        runtime = $runtime
        model_format = $modelFormat
        decode_strategy = $decodeStrategy
        max_context = $maxContext
    }
}

function Write-SharedResultRow($Path, [object]$Row) {
    $columns = @(
        "timestamp_utc", "requested_backend", "resolved_backend", "device", "device_name", "device_full_name",
        "model_package", "variant_id", "base_model", "precision", "quant_method", "execution_provider",
        "model_dir", "audio_path", "audio_seconds", "runs", "warmup", "cache_dir",
        "cold_load_seconds", "warm_load_seconds", "mean_infer_seconds", "rtf", "realtime_factor",
        "label", "model_size_mb", "avg_logprob", "ttft_ms", "tpot_ms", "throughput_tps",
        "wer", "cer", "transcription",
        "runtime", "model_format", "decode_strategy", "max_context", "eval_clips", "status",
        "cold_start_seconds", "hot_start_seconds", "power_source"
    )

    $parent = Split-Path $Path -Parent
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $header = $columns -join ","
    if (-not (Test-Path $Path) -or (Get-Item $Path).Length -eq 0) {
        $header | Set-Content -Path $Path -Encoding UTF8
    } else {
        $existingHeader = Get-Content -Path $Path -TotalCount 1
        if ($existingHeader -ne $header) {
            $oldRows = @(Import-Csv -Path $Path)
            $header | Set-Content -Path $Path -Encoding UTF8
            foreach ($old in $oldRows) {
                $oldLine = ($columns | ForEach-Object { ConvertTo-CsvCell (Get-Optional $old $_ "") }) -join ","
                Add-Content -Path $Path -Value $oldLine -Encoding UTF8
            }
        }
    }

    $line = ($columns | ForEach-Object { ConvertTo-CsvCell (Get-Optional $Row $_ "") }) -join ","
    Add-Content -Path $Path -Value $line -Encoding UTF8
}

function Invoke-Clip($modelDir, $audio, $backend, $device, $runs, $cacheDir, $ref, $label) {
    $a = @($modelDir, $audio, $backend, $device, "$runs", "--cache", $cacheDir, "--ref", $ref, "--label", $label, "--json")
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & $exe @a 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldEap
    }
    $line = ($raw | Where-Object { $_ -match '^\{"ok"' } | Select-Object -Last 1)
    if (-not $line) { return [pscustomobject]@{ ok = $false; error = "no-json-output (exit $exitCode)" } }
    return $line | ConvertFrom-Json
}

$summary = @()
$detail = @()

foreach ($v in $manifestObj.variants) {
    $modelDir = Join-Path $root ($v.model_dir -replace '/', '\')
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
        $coldLoad = $null; $hotLoad = $null

        if (-not (Test-Path $modelDir)) {
            $status = "missing-model"
            $errMsg = "model_dir missing ($modelDir)"
            Write-Host "   ! $errMsg" -ForegroundColor Yellow
        } else {
            foreach ($c in $clips) {
                $audio = Join-Path $root ($c.audio -replace '/', '\')
                if (-not (Test-Path $audio)) { continue }
                $r = Invoke-Clip $modelDir $audio $v.backend $dev $Runs $cacheDir $c.ref $v.id
                if (-not $r.ok) {
                    $status = "unsupported/error"; $errMsg = $r.error
                    Write-Host ("   {0}: {1}" -f $c.id, $r.error) -ForegroundColor Yellow
                    break
                }
                if ($first) {
                    $coldLoad = $r.load_cold_s
                    $hotLoad = if ($r.PSObject.Properties.Name -contains "load_hot_s") { $r.load_hot_s } else { $r.load_warm_s }
                    $first = $false
                }
                $rows += $r
                $detail += [pscustomobject]@{
                    variant = $v.id; precision = $v.precision; backend = $v.backend; device = $dev
                    clip = $c.id; mean_ms = [math]::Round($r.mean_ms, 1); rtf = [math]::Round($r.rtf, 4)
                    avg_logprob = [math]::Round($r.avg_logprob, 4); wer = [math]::Round($r.wer * 100, 2)
                    cer = [math]::Round($r.cer * 100, 2)
                }
            }
        }

        if ($rows.Count -eq 0) {
            $summary += [pscustomobject]@{
                variant = $v.id; precision = $v.precision; backend = $v.backend; device = $dev
                status = $status; clips = 0; size_mb = $v.size_mb
                cold_s = $null; hot_s = $null; mean_ms = $null; rtf = $null; xrt = $null
                tps = $null; avg_logprob = $null; wer_pct = $null; cer_pct = $null; error = $errMsg
            }
            Write-SharedResultRow $Results ([pscustomobject]@{
                timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
                requested_backend = $v.backend; resolved_backend = ""; device = $dev; device_name = ""; device_full_name = ""
                model_package = (Get-BenchmarkMeta $v $v.model_dir $null).model_package
                variant_id = $v.id; base_model = $manifestObj.model; precision = $v.precision; quant_method = $v.method
                execution_provider = ""
                model_dir = $v.model_dir; audio_path = $EvalSet; audio_seconds = ""; runs = $Runs; warmup = 1; cache_dir = $cacheDir
                cold_load_seconds = ""; warm_load_seconds = ""; mean_infer_seconds = ""; rtf = ""; realtime_factor = ""
                label = $v.id; model_size_mb = $v.size_mb; avg_logprob = ""; ttft_ms = ""; tpot_ms = ""; throughput_tps = ""
                wer = ""; cer = ""; transcription = ""; cold_start_seconds = ""; hot_start_seconds = ""; eval_clips = 0
                status = "$(if ($status -eq 'ok') { 'fail' } else { $status })$(if ($errMsg) { " ($errMsg)" })"
                runtime = ""; model_format = ""; decode_strategy = ""; max_context = ""; power_source = (Get-PowerSource)
            })
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
        $audioSeconds = ($rows | Measure-Object audio_len_s -Sum).Sum
        $firstRow = $rows | Select-Object -First 1
        $lastRow = $rows | Select-Object -Last 1
        $wer = if ($wRef) { $wEdits / $wRef } else { $null }
        $cer = if ($cRef) { $cEdits / $cRef } else { $null }

        $summary += [pscustomobject]@{
            variant = $v.id; precision = $v.precision; backend = $v.backend; device = $dev
            status = "ok"; clips = $rows.Count; size_mb = $v.size_mb
            cold_s = [math]::Round($coldLoad, 2); hot_s = [math]::Round($hotLoad, 2)
            mean_ms = [math]::Round($meanMs, 1); rtf = [math]::Round($meanRtf, 4)
            xrt = [math]::Round((1.0 / [math]::Max($meanRtf, 1e-9)), 1); tps = [math]::Round($meanTps, 1)
            avg_logprob = [math]::Round($meanLp, 4)
            wer_pct = if ($wRef) { [math]::Round(100.0 * $wer, 2) } else { $null }
            cer_pct = if ($cRef) { [math]::Round(100.0 * $cer, 2) } else { $null }
            error = ""
        }
        $meta = Get-BenchmarkMeta $v $v.model_dir $firstRow
        Write-SharedResultRow $Results ([pscustomobject]@{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            requested_backend = $v.backend; resolved_backend = $firstRow.backend; device = $dev
            device_name = $firstRow.device; device_full_name = $firstRow.device_full_name
            model_package = $meta.model_package; variant_id = $meta.variant_id; base_model = $meta.base_model
            precision = $meta.precision; quant_method = $meta.quant_method; execution_provider = $meta.execution_provider
            model_dir = $v.model_dir; audio_path = $EvalSet; audio_seconds = $audioSeconds; runs = $Runs; warmup = 1; cache_dir = $cacheDir
            cold_load_seconds = $coldLoad; warm_load_seconds = $hotLoad
            mean_infer_seconds = ($meanMs / 1000.0); rtf = $meanRtf
            realtime_factor = if ($meanRtf -gt 0) { 1.0 / $meanRtf } else { "" }
            label = $meta.variant_id; model_size_mb = $v.size_mb; avg_logprob = $meanLp
            ttft_ms = ($rows | Measure-Object ttft_ms -Average).Average
            tpot_ms = ($rows | Measure-Object tpot_ms -Average).Average
            throughput_tps = $meanTps; wer = $wer; cer = $cer; transcription = $lastRow.text
            cold_start_seconds = $coldLoad; hot_start_seconds = $hotLoad; eval_clips = $rows.Count; status = "ok"
            runtime = $meta.runtime; model_format = $meta.model_format; decode_strategy = $meta.decode_strategy
            max_context = $meta.max_context
            power_source = $(if (($firstRow.PSObject.Properties.Name -contains 'power_source') -and $firstRow.power_source) { $firstRow.power_source } else { Get-PowerSource })
        })
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
[void]$md.AppendLine("- Cold start = first engine creation after cache deletion; hot start = second engine creation in the same process after cache population.")
[void]$md.AppendLine("")
[void]$md.AppendLine("| Variant | Prec | Backend | Device | Status | Size MB | Cold s | Hot s | Mean ms | xRT | tok/s | Conf | WER % | CER % |")
[void]$md.AppendLine("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
foreach ($s in $summary) {
    $f = { param($x) if ($null -eq $x) { "-" } else { $x } }
    [void]$md.AppendLine(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} | {13} |" -f `
                $s.variant, $s.precision, $s.backend, $s.device, $s.status, (& $f $s.size_mb),
            (& $f $s.cold_s), (& $f $s.hot_s), (& $f $s.mean_ms), (& $f $s.xrt), (& $f $s.tps),
            (& $f $s.avg_logprob), (& $f $s.wer_pct), (& $f $s.cer_pct)))
}
[void]$md.AppendLine("")
[void]$md.AppendLine("_Caching: cold = first compile, hot = same-process reload from populated cache. See ``cache/`` and backend-specific compiled-model cache hooks._")
Set-Content -Path $mdPath -Value $md.ToString() -Encoding UTF8

Write-Host "`nReports written:" -ForegroundColor Green
Write-Host "  $csvPath"
Write-Host "  $detailCsv"
Write-Host "  $mdPath"
Write-Host ""
$summary | Format-Table variant, precision, device, status, size_mb, cold_s, hot_s, mean_ms, xrt, avg_logprob, wer_pct -AutoSize
