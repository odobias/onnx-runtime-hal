# Quantization RESEARCH sweep (formerly benchmark.ps1): runs every
# (variant x device x clip) from the manifest against the labeled eval set and produces
# CSV + Markdown reports comparing performance, self-confidence, and accuracy (WER/CER)
# per quantization method. This is NOT the default benchmark -- that is the portable
# benchmark-onnx.ps1 (C++ app). Use this to sweep OV-IR precision variants (fp16/int8/int4).
#
# Backend-agnostic by construction: it reads `backend` from each manifest entry and
# passes it straight to the runner, so AMD/Qualcomm variants drop in with no changes
# here (it autodetects the host NPU vendor and skips other vendors' variants). Unsupported
# (variant x device) combos are recorded, not fatal.
#
#   .\benchmark-quant.ps1                       # all variants, their listed devices
#   .\benchmark-quant.ps1 -Devices NPU          # restrict devices
#   .\benchmark-quant.ps1 -Runs 5 -MaxClips 10

[CmdletBinding()]
param(
    [string]$Manifest = "",
    [string]$EvalSet = "",
    [string[]]$Devices = @(),      # empty = use each variant's own device list
    [int]$Runs = 3,
    [int]$MaxClips = 0,            # 0 = all clips in the eval set
    [string]$Results = "",
    # NPU vendor filter: auto (detect this host), all (try everything), or a forced
    # vendor. A machine has one NPU brand, so by default we skip other vendors' variants.
    [ValidateSet("auto", "all", "intel", "amd", "qualcomm")][string]$NpuVendor = "auto",
    # Restrict the sweep to specific manifest variant id(s). Empty = all variants.
    [string[]]$Variant = @(),
    [string]$Configuration = "Release",
    [ValidateSet("x64", "ARM64")][string]$BuildPlatform = "",
    # Skip the self-contained bootstrap (build + fetch artifacts/workloads/speech). Use when
    # you have already prepared the environment and want the sweep to start faster.
    [switch]$SkipBootstrap,
    # Reuse any persisted compiled-model cache under cache/<variant>-<device> instead
    # of wiping it before each variant. This skips the ~215s NPU compile on reruns, but
    # cold_start_seconds then measures a load from the populated cache, NOT a true cold
    # compile. Default (off) keeps the honest cold-compile measurement.
    [switch]$ReuseCache,
    # Hot-only: skip the cold compile entirely and load each engine once from the
    # persisted cache (implies -ReuseCache). cold_start is reported as N/A. If a
    # variant/device has no cache yet, it is warmed once (one compile) before measuring.
    [switch]$HotOnly
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
. (Join-Path $PSScriptRoot "..\lib\harness.ps1")
Initialize-BenchmarkConsole

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $Manifest) { $Manifest = Join-Path $root "artifacts\workloads\whisper\manifest.research.json" }
if (-not $EvalSet) { $EvalSet = Join-Path $root "src\workloads\eval\eval.jsonl" }
if (-not $BuildPlatform) {
    $BuildPlatform = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { "ARM64" } else { "x64" }
}
$exe = Join-Path $root "artifacts\build\$BuildPlatform\$Configuration\NpuInferenceBench.exe"

# Detect the host once, up front: bootstrap needs it to pick build flags + models,
# and the sweep reuses it for the vendor filter and the hardware banner.
$platform = Get-BenchmarkPlatform

# Self-contained bootstrap: build the app for this host's backends and fetch any
# missing artifacts/workloads/speech so the benchmark runs from a fresh checkout. Idempotent
# (present artifacts are skipped) and opt-out via -SkipBootstrap.
if (-not $SkipBootstrap) {
    Initialize-BenchmarkEnvironment -Root $root -Exe $exe -Manifest $Manifest -EvalSet $EvalSet `
        -Configuration $Configuration -Platform $platform
}

foreach ($p in @($Manifest, $EvalSet, $exe)) {
    if (-not (Test-Path $p)) {
        Write-Host "Missing: $p (run without -SkipBootstrap, or .\tools\build\bootstrap.ps1)" -ForegroundColor Red
        exit 1
    }
}

$manifestObj = Get-Content $Manifest -Raw | ConvertFrom-Json
if ($Variant.Count) {
    $manifestObj.variants = @($manifestObj.variants | Where-Object { $Variant -contains $_.id })
    if (-not $manifestObj.variants.Count) { Write-Host "No manifest variant matched: $($Variant -join ', ')" -ForegroundColor Red; exit 1 }
}
$clips = Get-Content $EvalSet | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
if ($MaxClips -gt 0 -and $clips.Count -gt $MaxClips) { $clips = $clips[0..($MaxClips - 1)] }
Write-Host ("Manifest: {0} variant(s) | Eval: {1} clip(s) | Runs: {2}" -f `
        $manifestObj.variants.Count, $clips.Count, $Runs) -ForegroundColor Cyan

$reportsDir = Join-Path $root "artifacts\scratch"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
if (-not $Results) { $Results = Join-Path $reportsDir "benchmark-results.csv" }
$cacheRoot = Join-Path $root "artifacts\cache"

# Autodetect the host NPU vendor (one brand per machine) and skip variants that
# target a different vendor's NPU. -NpuVendor all disables the filter; -NpuVendor
# <intel|amd|qualcomm> forces it. Shared harness helpers (CSV schema, power source,
# app invocation, aggregation, detection) live in benchmark/lib/harness.ps1.
$hostVendor = Resolve-BenchmarkHostVendor $NpuVendor $platform

# Detailed CPU/GPU/NPU inventory of the machine these results were produced on.
$hardware = Get-BenchmarkHardware
Write-BenchmarkHardwareBanner $hardware

$summary = @()
$detail = @()

foreach ($v in $manifestObj.variants) {
    if (-not (Test-BenchmarkVariantSupported -Backend $v.backend -HostVendor $hostVendor)) {
        $vend = Get-BenchmarkBackendVendor $v.backend
        Write-Host ("-- skip {0} (backend {1} targets {2} NPU; host is {3}) --" -f `
                $v.id, $v.backend, $vend, $hostVendor) -ForegroundColor DarkGray
        $summary += [pscustomobject]@{
            variant = $v.id; precision = $v.precision; backend = $v.backend; device = "-"
            status = "skip:other-npu"; clips = 0; size_mb = $v.size_mb
            cold_s = $null; hot_s = $null; mean_ms = $null; rtf = $null; xrt = $null
            tps = $null; avg_logprob = $null; wer_pct = $null; cer_pct = $null
            error = "$vend variant on $hostVendor host"
        }
        continue
    }
    $modelDir = Join-Path $root ($v.model_dir -replace '/', '\')
    $devList = if ($Devices.Count) { $Devices } else { $v.devices }
    # Neutral self-selecting variants run through the unified OVEP binary on Intel
    # (the default GenAI exe has no ORT backend); all others use the default exe.
    $variantExe = Get-BenchmarkExeForVariant $v $exe $root $hostVendor $Configuration
    if ($variantExe -ne $exe) {
        Write-Host ("   (via unified OVEP binary: {0})" -f (Split-Path $variantExe -Leaf)) -ForegroundColor DarkCyan
    }

    foreach ($dev in $devList) {
        $tag = "$($v.id)-$dev"
        Write-Host "== $tag ==" -ForegroundColor White
        $cacheDir = Join-Path $cacheRoot $tag
        if ($ReuseCache -or $HotOnly) {
            if (Test-Path $cacheDir) {
                Write-Host "   (reusing persisted cache; cold_start reflects cached load, not a true compile)" -ForegroundColor DarkYellow
            }
        }
        elseif (Test-Path $cacheDir) { Remove-Item $cacheDir -Recurse -Force }  # force a true cold compile

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
            # Hot-only needs a populated cache; warm it with one compile if it is empty.
            if ($HotOnly) {
                $populated = (Test-Path $cacheDir) -and `
                    ((Get-ChildItem -Path $cacheDir -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1) -ne $null)
                if (-not $populated) {
                    $warm = $clips | Where-Object { Test-Path (Join-Path $root ($_.audio -replace '/', '\')) } | Select-Object -First 1
                    if ($warm) {
                        Write-Host "   (hot-only: cache empty -> warming with one cold compile)" -ForegroundColor DarkYellow
                        $warmAudio = Join-Path $root ($warm.audio -replace '/', '\')
                        $null = Invoke-BenchmarkClip -Exe $variantExe -ModelDir $modelDir -Audio $warmAudio -Backend $v.backend -Device $dev -Runs 1 -CacheDir $cacheDir -Ref $warm.ref
                    }
                }
            }
            foreach ($c in $clips) {
                $audio = Join-Path $root ($c.audio -replace '/', '\')
                if (-not (Test-Path $audio)) { continue }
                $r = Invoke-BenchmarkClip -Exe $variantExe -ModelDir $modelDir -Audio $audio -Backend $v.backend -Device $dev -Runs $Runs -CacheDir $cacheDir -Ref $c.ref -HotOnly:$HotOnly
                if (-not $r.ok) {
                    $status = "unsupported/error"; $errMsg = $r.error
                    Write-Host ("   {0}: {1}" -f $c.id, $r.error) -ForegroundColor Yellow
                    break
                }
                if ($first) {
                    # Negative cold load = N/A (hot-only did not compile).
                    $coldLoad = if ($r.load_cold_s -lt 0) { $null } else { $r.load_cold_s }
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
            $meta = Get-BenchmarkMeta $v $v.model_dir $null $manifestObj.model
            Write-BenchmarkResultRow $Results ([pscustomobject]@{
                timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
                requested_backend = $v.backend; resolved_backend = ""; device = $dev; device_name = ""; device_full_name = ""
                model_package = $meta.model_package; model_sha256 = ""; variant_id = $meta.variant_id; base_model = $meta.base_model
                precision = $meta.precision; quant_method = $meta.quant_method; execution_provider = ""
                model_dir = $v.model_dir; audio_path = $EvalSet; audio_seconds = ""; runs = $Runs; warmup = 1; cache_dir = $cacheDir
                cold_load_seconds = ""; warm_load_seconds = ""; mean_infer_seconds = ""; rtf = ""; realtime_factor = ""
                label = $v.id; model_size_mb = $v.size_mb; avg_logprob = ""; ttft_ms = ""; tpot_ms = ""; throughput_tps = ""
                wer = ""; cer = ""; transcription = ""; cold_start_seconds = ""; hot_start_seconds = ""; eval_clips = 0
                status = "$(if ($status -eq 'ok') { 'fail' } else { $status })$(if ($errMsg) { " ($errMsg)" })"
                runtime = ""; model_format = ""; decode_strategy = ""; max_context = ""; power_source = (Get-BenchmarkPowerSource)
                host_arch = (Get-BenchmarkHostArch); host_os = (Get-BenchmarkHostOs); runtime_version = ""
            })
            continue
        }

        $agg = Measure-BenchmarkClips $rows
        $meanMs = $agg.mean_ms; $meanRtf = $agg.mean_rtf; $meanTps = $agg.mean_tps; $meanLp = $agg.mean_logprob
        $wEdits = $agg.word_edits; $wRef = $agg.ref_words; $cEdits = $agg.char_edits; $cRef = $agg.ref_chars
        $audioSeconds = $agg.audio_seconds; $firstRow = $agg.first_row; $lastRow = $agg.last_row
        $wer = $agg.wer; $cer = $agg.cer

        $summary += [pscustomobject]@{
            variant = $v.id; precision = $v.precision; backend = $v.backend; device = $dev
            status = "ok"; clips = $rows.Count; size_mb = $v.size_mb
            cold_s = if ($null -eq $coldLoad) { $null } else { [math]::Round($coldLoad, 2) }
            hot_s = if ($null -eq $hotLoad) { $null } else { [math]::Round($hotLoad, 2) }
            mean_ms = [math]::Round($meanMs, 1); rtf = [math]::Round($meanRtf, 4)
            xrt = [math]::Round((1.0 / [math]::Max($meanRtf, 1e-9)), 1); tps = [math]::Round($meanTps, 1)
            avg_logprob = [math]::Round($meanLp, 4)
            wer_pct = if ($wRef) { [math]::Round(100.0 * $wer, 2) } else { $null }
            cer_pct = if ($cRef) { [math]::Round(100.0 * $cer, 2) } else { $null }
            error = ""
        }
        $meta = Get-BenchmarkMeta $v $v.model_dir $firstRow $manifestObj.model
        Write-BenchmarkResultRow $Results ([pscustomobject]@{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            requested_backend = $v.backend; resolved_backend = $firstRow.backend; device = $dev
            device_name = $firstRow.device; device_full_name = $firstRow.device_full_name
            model_package = $meta.model_package
            model_sha256 = (Get-BenchmarkRowValue $firstRow 'model_sha256' '')
            variant_id = $meta.variant_id; base_model = $meta.base_model
            precision = $meta.precision; quant_method = $meta.quant_method; execution_provider = $meta.execution_provider
            model_dir = $v.model_dir; audio_path = $EvalSet; audio_seconds = $audioSeconds; runs = $Runs; warmup = 1; cache_dir = $cacheDir
            cold_load_seconds = $coldLoad; warm_load_seconds = $hotLoad
            mean_infer_seconds = ($meanMs / 1000.0); rtf = $meanRtf
            realtime_factor = if ($meanRtf -gt 0) { 1.0 / $meanRtf } else { "" }
            label = $meta.variant_id; model_size_mb = $v.size_mb; avg_logprob = $meanLp
            ttft_ms = $agg.mean_ttft_ms
            tpot_ms = $agg.mean_tpot_ms
            throughput_tps = $meanTps; wer = $wer; cer = $cer; transcription = $lastRow.text
            cold_start_seconds = $coldLoad; hot_start_seconds = $hotLoad; eval_clips = $rows.Count; status = "ok"
            runtime = $meta.runtime; model_format = $meta.model_format; decode_strategy = $meta.decode_strategy
            max_context = $meta.max_context
            power_source = $(if (($firstRow.PSObject.Properties.Name -contains 'power_source') -and $firstRow.power_source) { $firstRow.power_source } else { Get-BenchmarkPowerSource })
            host_arch = (Get-BenchmarkRowValue $firstRow 'host_arch' (Get-BenchmarkHostArch))
            host_os = (Get-BenchmarkRowValue $firstRow 'host_os' (Get-BenchmarkHostOs))
            runtime_version = (Get-BenchmarkRowValue $firstRow 'runtime_version' '')
            inference_precision = (Get-BenchmarkRowValue $firstRow 'inference_precision' '')
            ep_nodes = (Get-BenchmarkRowValue $firstRow 'ep_nodes' '')
            cpu_nodes = (Get-BenchmarkRowValue $firstRow 'cpu_nodes' '')
            cpu_offload_pct = (Get-BenchmarkRowValue $firstRow 'cpu_offload_pct' '')
            cpu_offload_ops = (Get-BenchmarkRowValue $firstRow 'cpu_offload_ops' '')
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
[void]$md.AppendLine("# NPU Inference Benchmark - quantization benchmark")
[void]$md.AppendLine("")
[void]$md.AppendLine("- Model: ``$($manifestObj.model)``")
[void]$md.AppendLine("- Host: ``$($platform.cpu_name)`` | NPU vendor ``$($platform.npu_vendor)``$(if ($platform.npu_device) { " ($($platform.npu_device))" }) | filter ``$hostVendor``")
[void]$md.AppendLine("- Eval clips: $($clips.Count) | Runs/clip: $Runs | Generated: $(Get-Date -Format s)")
[void]$md.AppendLine("- Confidence = mean per-token log-prob (self-reported; higher = more confident, not calibrated truth).")
[void]$md.AppendLine("- WER/CER micro-averaged over clips after normalization.")
[void]$md.AppendLine("- Cold start = first engine creation after cache deletion; hot start = second engine creation in the same process after cache population.")
[void]$md.AppendLine("")
[void]$md.AppendLine("## Host hardware")
[void]$md.AppendLine("")
foreach ($line in (Get-BenchmarkHardwareMarkdown $hardware)) { [void]$md.AppendLine($line) }
[void]$md.AppendLine("")
[void]$md.AppendLine("_Full machine-readable inventory: ``artifacts/scratch/host-info.json``._")
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
$hostInfoPath = Join-Path $reportsDir "host-info.json"
Write-BenchmarkHostInfo $hostInfoPath $hardware

Write-Host "`nReports written:" -ForegroundColor Green
Write-Host "  $csvPath"
Write-Host "  $detailCsv"
Write-Host "  $mdPath"
Write-Host "  $hostInfoPath"
Write-Host ""
$summary | Format-Table variant, precision, device, status, size_mb, cold_s, hot_s, mean_ms, xrt, avg_logprob, wer_pct -AutoSize
