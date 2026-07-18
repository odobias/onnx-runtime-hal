# Device comparison harness: runs ONE model variant across NPU / GPU / CPU and
# produces a device-pivoted report (latency, RTF, tok/s, confidence, WER/CER, plus
# a speedup-vs-CPU column). Complements benchmark-quant.ps1 (which pivots on quantization
# variant); this pivots on device so "is the NPU worth it vs. just running on CPU"
# is a single glance instead of grepping a variant x device matrix.
#
# Unsupported combos (no NPU/Intel-GPU present, model too small to compile, etc.)
# are recorded as failures, not fatal -- same backend-neutral contract as
# benchmark-quant.ps1. Reads/writes the same manifest.json + eval.jsonl.
#
#   .\compare-devices.ps1                                # fp16 variant, NPU/GPU/CPU
#   .\compare-devices.ps1 -Variant wten-ov-int8
#   .\compare-devices.ps1 -Devices CPU -Threads 8         # CPU-only, pinned threads
#   .\compare-devices.ps1 -Devices NPU,CPU -MaxClips 5

[CmdletBinding()]
param(
    [string]$Manifest = "",
    [string]$EvalSet = "",
    [string]$Variant = "",           # manifest variant id; empty = first fp16, else first entry
    [string[]]$Devices = @("NPU", "GPU", "CPU"),
    [int]$Runs = 3,
    [int]$Threads = 0,                # CPU only; 0 = runtime default
    [int]$MaxClips = 0,               # 0 = all clips in the eval set
    # NPU vendor: auto (detect host, prefer matching variant), all (no preference),
    # or a forced vendor. Only affects default variant selection + a mismatch warning.
    [ValidateSet("auto", "all", "intel", "amd", "qualcomm")][string]$NpuVendor = "auto",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
. (Join-Path $PSScriptRoot "lib\harness.ps1")
Initialize-BenchmarkConsole

$root = Split-Path $PSScriptRoot -Parent
if (-not $Manifest) { $Manifest = Join-Path $root "src\workloads\whisper\manifest.json" }
if (-not $EvalSet) { $EvalSet = Join-Path $root "src\workloads\eval\eval.jsonl" }
$exe = Join-Path $root "build\x64\$Configuration\NpuInferenceBench.exe"

foreach ($p in @($Manifest, $EvalSet, $exe)) {
    if (-not (Test-Path $p)) { Write-Host "Missing: $p" -ForegroundColor Red; exit 1 }
}

$manifestObj = Get-Content $Manifest -Raw | ConvertFrom-Json
if (-not $manifestObj.variants -or $manifestObj.variants.Count -eq 0) {
    Write-Host "Manifest has no variants. Run export-variants.ps1 first." -ForegroundColor Red
    exit 1
}

# Autodetect the host NPU vendor so default variant selection prefers something this
# machine can actually run (one NPU brand per host).
$platform = Get-BenchmarkPlatform
$hostVendor = Resolve-BenchmarkHostVendor $NpuVendor $platform
$hardware = Get-BenchmarkHardware
Write-BenchmarkHardwareBanner $hardware
$supported = @($manifestObj.variants | Where-Object { Test-BenchmarkVariantSupported -Backend $_.backend -HostVendor $hostVendor })

$v = $null
if ($Variant) {
    $v = $manifestObj.variants | Where-Object { $_.id -eq $Variant } | Select-Object -First 1
    if (-not $v) { Write-Host "Variant '$Variant' not found in manifest." -ForegroundColor Red; exit 1 }
    if (-not (Test-BenchmarkVariantSupported -Backend $v.backend -HostVendor $hostVendor)) {
        Write-Host ("  ! '{0}' targets {1} NPU but host is {2}; running anyway as requested." -f `
                $v.id, (Get-BenchmarkBackendVendor $v.backend), $hostVendor) -ForegroundColor Yellow
    }
} else {
    $pool = if ($supported.Count) { $supported } else { $manifestObj.variants }
    $v = $pool | Where-Object { $_.precision -eq "fp16" } | Select-Object -First 1
    if (-not $v) { $v = $pool | Select-Object -First 1 }
}

$modelDir = Join-Path $root ($v.model_dir -replace '/', '\')
if (-not (Test-Path $modelDir)) { Write-Host "model_dir missing: $modelDir" -ForegroundColor Red; exit 1 }

$clips = Get-Content $EvalSet | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
if ($MaxClips -gt 0 -and $clips.Count -gt $MaxClips) { $clips = $clips[0..($MaxClips - 1)] }

Write-Host ("Variant: {0} ({1}, {2} MB, backend {3}) | Devices: {4} | Eval clips: {5} | Runs/clip: {6}" -f `
        $v.id, $v.precision, $v.size_mb, $v.backend, ($Devices -join ", "), $clips.Count, $Runs) -ForegroundColor Cyan

$reportsDir = Join-Path $root "build\reports"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
$cacheRoot = Join-Path $root "cache"

# All devices run in this one session, so a single power reading applies to the
# whole comparison. Power source + clip invocation come from benchmark/lib/harness.ps1.
$powerSource = Get-BenchmarkPowerSource
Write-Host ("Power: {0}" -f $powerSource) -ForegroundColor Cyan

$rows = @()
foreach ($dev in $Devices) {
    Write-Host "== $dev ==" -ForegroundColor White
    $cacheDir = Join-Path $cacheRoot "compare-$($v.id)-$dev"
    if (Test-Path $cacheDir) { Remove-Item $cacheDir -Recurse -Force }  # force a true cold compile

    $clipResults = @()
    $status = "ok"; $errMsg = ""
    $coldLoad = $null; $hotLoad = $null; $threadsUsed = $null; $hwConcurrency = $null
    $chip = $null
    $first = $true

    foreach ($c in $clips) {
        $audio = Join-Path $root ($c.audio -replace '/', '\')
        if (-not (Test-Path $audio)) { continue }
        $r = Invoke-BenchmarkClip -Exe $exe -ModelDir $modelDir -Audio $audio -Backend $v.backend -Device $dev -Runs $Runs -CacheDir $cacheDir -Ref $c.ref -Threads $Threads
        if (-not $r.ok) {
            $status = "unsupported/error"
            $errMsg = ($r.error -split "`n")[0]
            Write-Host ("   ! {0}" -f $errMsg) -ForegroundColor Yellow
            break
        }
        if ($first) {
            $load = Get-BenchmarkLoadTimes $r
            $coldLoad = $load.cold; $hotLoad = $load.hot
            $threadsUsed = $r.cpu_threads_requested; $hwConcurrency = $r.hw_concurrency
            $chip = $r.device_full_name
            $first = $false
        }
        $clipResults += $r
    }

    if ($clipResults.Count -eq 0) {
        $rows += [pscustomobject]@{
            device = $dev; chip = $null; status = $status; clips = 0
            cold_s = $null; hot_s = $null; mean_ms = $null; rtf = $null; xrt = $null
            tps = $null; avg_logprob = $null; wer_pct = $null; cer_pct = $null
            threads = $null; hw_concurrency = $null; power_source = $powerSource; error = $errMsg
        }
        continue
    }

    $agg = Measure-BenchmarkClips $clipResults
    $meanMs = $agg.mean_ms; $meanRtf = $agg.mean_rtf; $meanTps = $agg.mean_tps; $meanLp = $agg.mean_logprob
    $wEdits = $agg.word_edits; $wRef = $agg.ref_words; $cEdits = $agg.char_edits; $cRef = $agg.ref_chars

    $rows += [pscustomobject]@{
        device = $dev; chip = $(if ($chip) { $chip } else { "-" }); status = "ok"; clips = $clipResults.Count
        cold_s = [math]::Round($coldLoad, 3); hot_s = [math]::Round($hotLoad, 3)
        mean_ms = [math]::Round($meanMs, 1); rtf = [math]::Round($meanRtf, 4)
        xrt = [math]::Round((1.0 / [math]::Max($meanRtf, 1e-9)), 1); tps = [math]::Round($meanTps, 1)
        avg_logprob = [math]::Round($meanLp, 4)
        wer_pct = if ($wRef) { [math]::Round(100.0 * $wEdits / $wRef, 2) } else { $null }
        cer_pct = if ($cRef) { [math]::Round(100.0 * $cEdits / $cRef, 2) } else { $null }
        threads = if ($dev -eq "CPU") { if ($threadsUsed -gt 0) { $threadsUsed } else { "default" } } else { "-" }
        hw_concurrency = if ($dev -eq "CPU") { $hwConcurrency } else { "-" }
        power_source = $powerSource; error = ""
    }
    Write-Host ("   ok: {0} clips | {1} ms | {2}x RT | WER {3}%" -f `
            $clipResults.Count, [math]::Round($meanMs, 1), [math]::Round((1.0 / [math]::Max($meanRtf, 1e-9)), 1),
        $(if ($wRef) { [math]::Round(100.0 * $wEdits / $wRef, 2) } else { "n/a" })) -ForegroundColor Green
}

# Speedup relative to the slowest *successful* device (usually CPU) -- makes the
# NPU/GPU value proposition (or lack thereof, on hardware where they're absent) explicit.
$okRows = $rows | Where-Object { $_.status -eq "ok" }
$slowestMs = if ($okRows) { ($okRows | Measure-Object mean_ms -Maximum).Maximum } else { $null }
foreach ($r in $rows) {
    $speedup = if ($r.status -eq "ok" -and $slowestMs) { [math]::Round($slowestMs / $r.mean_ms, 2) } else { $null }
    $r | Add-Member -NotePropertyName speedup -NotePropertyValue $speedup
}

$csvPath = Join-Path $reportsDir "device-compare.csv"
$mdPath = Join-Path $reportsDir "device-compare.md"
$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("# NPU Inference Benchmark - device comparison (NPU vs GPU vs CPU)")
[void]$md.AppendLine("")
[void]$md.AppendLine("- Model: ``$($manifestObj.model)`` | Variant: ``$($v.id)`` ($($v.precision), $($v.size_mb) MB, backend ``$($v.backend)``)")
[void]$md.AppendLine("- Eval clips: $($clips.Count) | Runs/clip: $Runs | Generated: $(Get-Date -Format s)")
[void]$md.AppendLine("- Power source: ``$powerSource`` (battery = throttled clocks; treat cross-machine numbers accordingly).")
[void]$md.AppendLine("- Speedup = slowest-successful-device / this device's mean latency (bigger = faster).")
[void]$md.AppendLine("- Confidence = mean per-token log-prob (self-reported, not calibrated truth).")
[void]$md.AppendLine("- Cold start = first engine creation after cache deletion; hot start = second engine creation in the same process after cache population.")
[void]$md.AppendLine("")
[void]$md.AppendLine("## Host hardware")
[void]$md.AppendLine("")
foreach ($line in (Get-BenchmarkHardwareMarkdown $hardware)) { [void]$md.AppendLine($line) }
[void]$md.AppendLine("")
[void]$md.AppendLine("_Full machine-readable inventory: ``build/reports/host-info.json``._")
[void]$md.AppendLine("")
[void]$md.AppendLine("| Device | Chip | Status | Threads | Cold s | Hot s | Mean ms | xRT | Speedup | tok/s | Conf | WER % | CER % |")
[void]$md.AppendLine("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
foreach ($r in $rows) {
    $f = { param($x) if ($null -eq $x -or $x -eq "") { "-" } else { $x } }
    $threadsCol = if ($r.device -eq "CPU") { "$($r.threads)/$($r.hw_concurrency)" } else { "-" }
    [void]$md.AppendLine(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} |" -f `
                $r.device, (& $f $r.chip), $r.status, $threadsCol, (& $f $r.cold_s), (& $f $r.hot_s), (& $f $r.mean_ms),
            (& $f $r.xrt), (& $f $r.speedup), (& $f $r.tps), (& $f $r.avg_logprob), (& $f $r.wer_pct), (& $f $r.cer_pct)))
}
[void]$md.AppendLine("")
foreach ($r in $rows) {
    if ($r.status -ne "ok" -and $r.error) {
        [void]$md.AppendLine("- **$($r.device)** failed: ``$($r.error)``")
    }
}
Set-Content -Path $mdPath -Value $md.ToString() -Encoding UTF8
$hostInfoPath = Join-Path $reportsDir "host-info.json"
Write-BenchmarkHostInfo $hostInfoPath $hardware

Write-Host "`nReports written:" -ForegroundColor Green
Write-Host "  $csvPath"
Write-Host "  $mdPath"
Write-Host "  $hostInfoPath"
Write-Host ""
$rows | Format-Table device, chip, status, threads, cold_s, hot_s, mean_ms, xrt, speedup, wer_pct -AutoSize
