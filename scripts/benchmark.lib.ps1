# Shared benchmark harness library. This is the single home for the concerns that
# every benchmark entry point (benchmark-onnx.ps1, benchmark-quant.ps1, compare-devices.ps1, run.ps1) needs:
# the canonical results-CSV schema, the app invocation + JSON parse, clip aggregation
# math, and best-effort power-source detection.
#
# Dot-source it from a sibling script:
#     . (Join-Path $PSScriptRoot "benchmark.lib.ps1")
#
# Design: the C++ app (app/main.cpp) is the executor -- it runs ONE clip on ONE
# backend/device and emits one JSON record (and optionally appends one CSV row).
# This library is the harness -- it owns invocation, aggregation across clips, and
# writing the shared cross-machine ledger. Keep that split intact: aggregation and
# benchmark policy live here, not in the app.

# --- console -----------------------------------------------------------------

# Force UTF-8 across chcp / console / pipeline so non-ASCII transcripts survive
# capture. Call once near the top of an entry-point script.
function Initialize-BenchmarkConsole {
    chcp 65001 > $null
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
    $global:OutputEncoding = [System.Text.UTF8Encoding]::new()
}

# --- results CSV schema (single source of truth) -----------------------------

# Canonical column order for results/benchmark-results.csv. This is the authoritative
# list for the PowerShell harness; the C++ app (app/main.cpp) writes the same schema
# for its single-run rows and MUST be kept in sync with this list. See results/README.md.
function Get-BenchmarkResultColumns {
    # AUTHORITATIVE column order. Must stay byte-identical to kBenchmarkCsvHeader in
    # app/main.cpp (the other writer of this ledger). Change both + results/README.md.
    return @(
        "timestamp_utc", "requested_backend", "resolved_backend", "device", "device_name", "device_full_name",
        "model_package", "variant_id", "base_model", "precision", "quant_method", "execution_provider",
        "model_dir", "audio_path", "audio_seconds", "runs", "warmup", "cache_dir",
        "cold_load_seconds", "warm_load_seconds", "mean_infer_seconds", "rtf", "realtime_factor",
        "label", "model_size_mb", "avg_logprob", "ttft_ms", "tpot_ms", "throughput_tps",
        "wer", "cer", "transcription",
        "runtime", "model_format", "decode_strategy", "max_context", "eval_clips", "status",
        "cold_start_seconds", "hot_start_seconds", "power_source",
        "host_arch", "host_os", "runtime_version"
    )
}

# Resolve the forced, filterable metadata for a benchmark row. Mirrors
# benchmark_meta.hpp: derive model_package from model_dir, prefer manifest fields
# (precision/method/id), then overlay whatever the run's JSON already reported.
function Get-BenchmarkModelPackage([string]$ModelDir) {
    if (-not $ModelDir) { return "" }
    return [System.IO.Path]::GetFileName(($ModelDir -replace '\\', '/').TrimEnd('/'))
}

function Get-BenchmarkInferredPrecision([string]$Package) {
    $p = $Package.ToLowerInvariant()
    if ($p -match 'int4') { return 'int4' }
    if ($p -match 'int8') { return 'int8' }
    if ($p -match 'fp16' -or $p -match '-f16') { return 'fp16' }
    if ($p -match 'static') { return 'fp32-static' }
    return 'fp32'
}

function Get-BenchmarkMeta($Variant, $ModelDir, $JsonRow, $BaseModel) {
    $modelPackage = Get-BenchmarkModelPackage $ModelDir
    $variantId = if ($Variant -and $Variant.id) { $Variant.id } else { "" }
    if (-not $variantId -and $JsonRow) { $variantId = Get-BenchmarkOptional $JsonRow "variant_id" "" }
    if (-not $variantId) { $variantId = $modelPackage }

    $baseModel = if ($BaseModel) { $BaseModel } else { "openai/whisper-tiny.en" }
    $precision = if ($Variant -and $Variant.precision) { $Variant.precision } else { Get-BenchmarkInferredPrecision $modelPackage }
    $quantMethod = if ($Variant -and $Variant.method) { $Variant.method } else { "FP32 baseline (inferred from package name)" }

    # Backend-derived fields always come from the run's JSON. Manifest ($Variant) stays
    # authoritative for id/precision/method: several entries can share one model_dir
    # (e.g. the static ONNX run through ORT vs QNN vs OpenVINO), so trusting the exe's
    # by-model_dir manifest guess would mislabel them.
    $executionProvider = ""; $runtime = ""; $modelFormat = ""; $decodeStrategy = ""; $maxContext = ""
    $hasVariant = [bool]($Variant -and $Variant.id)
    if ($JsonRow) {
        $executionProvider = Get-BenchmarkOptional $JsonRow "execution_provider" (Get-BenchmarkOptional $JsonRow "device" "")
        $runtime = Get-BenchmarkOptional $JsonRow "runtime" (Get-BenchmarkOptional $JsonRow "backend" "")
        $modelFormat = Get-BenchmarkOptional $JsonRow "model_format" ""
        $decodeStrategy = Get-BenchmarkOptional $JsonRow "decode_strategy" ""
        $maxContext = Get-BenchmarkOptional $JsonRow "max_context" ""
        if (Get-BenchmarkOptional $JsonRow "base_model" "") { $baseModel = $JsonRow.base_model }
        if (-not $modelPackage -and (Get-BenchmarkOptional $JsonRow "model_package" "")) { $modelPackage = $JsonRow.model_package }
        if (-not $hasVariant) {
            if (Get-BenchmarkOptional $JsonRow "variant_id" "") { $variantId = $JsonRow.variant_id }
            if (Get-BenchmarkOptional $JsonRow "precision" "") { $precision = $JsonRow.precision }
            if (Get-BenchmarkOptional $JsonRow "quant_method" "") { $quantMethod = $JsonRow.quant_method }
        }
    }

    return [pscustomobject]@{
        model_package = $modelPackage; variant_id = $variantId; base_model = $baseModel
        precision = $precision; quant_method = $quantMethod; execution_provider = $executionProvider
        runtime = $runtime; model_format = $modelFormat; decode_strategy = $decodeStrategy; max_context = $maxContext
    }
}

function ConvertTo-BenchmarkCsvCell($Value) {
    $s = if ($null -eq $Value) { "" } else { [string]$Value }
    if ($s.IndexOfAny([char[]]",`"`r`n") -lt 0) { return $s }
    return '"' + ($s -replace '"', '""') + '"'
}

function Get-BenchmarkOptional($Object, [string]$Name, $Default = $null) {
    if ($null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
    return $Default
}

# Append one aggregate row to the shared results CSV, migrating an older/narrower
# header to the current schema in place (preserving existing rows) when needed.
function Write-BenchmarkResultRow($Path, [object]$Row) {
    $columns = Get-BenchmarkResultColumns

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
                $oldLine = ($columns | ForEach-Object { ConvertTo-BenchmarkCsvCell (Get-BenchmarkOptional $old $_ "") }) -join ","
                Add-Content -Path $Path -Value $oldLine -Encoding UTF8
            }
        }
    }

    $line = ($columns | ForEach-Object { ConvertTo-BenchmarkCsvCell (Get-BenchmarkOptional $Row $_ "") }) -join ","
    Add-Content -Path $Path -Value $line -Encoding UTF8
}

# --- power source ------------------------------------------------------------

# Best-effort AC vs battery detection (battery => throttled clocks, which silently
# skews comparisons). Returns "ac" / "battery" / "unknown".
function Get-BenchmarkPowerSource {
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        switch ([System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus) {
            'Online' { return 'ac' }
            'Offline' { return 'battery' }
            default { return 'unknown' }
        }
    } catch { return 'unknown' }
}

# --- host identity -----------------------------------------------------------

# ISA of the current host, normalized to the same tokens the C++ runner emits
# (x64 / arm64 / x86). This is the axis that decides which vendor DLL pack a build
# belongs to, so it's stamped per row to keep cross-ISA results attributable.
function Get-BenchmarkHostArch {
    try {
        switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
            'X64'   { return 'x64' }
            'Arm64' { return 'arm64' }
            'X86'   { return 'x86' }
            default { return "$([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)".ToLowerInvariant() }
        }
    } catch {
        $a = $env:PROCESSOR_ARCHITECTURE
        if ($a -eq 'AMD64') { return 'x64' }
        if ($a -eq 'ARM64') { return 'arm64' }
        if ($a -eq 'x86') { return 'x86' }
        return 'unknown'
    }
}

# Best-effort OS identity, e.g. "Windows 11 (build 26100)".
function Get-BenchmarkHostOs {
    try {
        $v = [Environment]::OSVersion.Version
        $name = if ($v.Major -eq 10 -and $v.Build -ge 22000) { 'Windows 11' }
                elseif ($v.Major -eq 10) { 'Windows 10' }
                else { 'Windows' }
        return "$name (build $($v.Build))"
    } catch { return 'Windows' }
}

# Prefer the value the runner binary reported (authoritative for "which build"),
# falling back to this host's own detection when the JSON row lacks it.
function Get-BenchmarkRowValue($JsonRow, [string]$Name, [string]$Fallback) {
    if ($JsonRow -and ($JsonRow.PSObject.Properties.Name -contains $Name) -and $JsonRow.$Name) {
        return $JsonRow.$Name
    }
    return $Fallback
}

# --- platform / NPU vendor autodetection -------------------------------------

# Map a manifest backend string to the NPU vendor it targets. "Neutral" backends
# (vendor-agnostic static ONNX, or Auto) run anywhere; vendor backends only run on
# a matching host. Keep this in sync with the backend aliases in app/main.cpp and
# the ValidateSet in run.ps1.
function Get-BenchmarkBackendVendor([string]$Backend) {
    switch -Regex ("$Backend".ToLowerInvariant()) {
        '^intel-onnx$' { return 'Intel' }
        '^intel$'      { return 'Intel' }
        '^amd'         { return 'AMD' }
        '^qualcomm'    { return 'Qualcomm' }
        default        { return 'Neutral' }  # onnx-static, ort, auto, unknown
    }
}

# Normalize a user-supplied vendor selector to the canonical casing used here.
function Get-BenchmarkVendorName([string]$Vendor) {
    switch -Regex ("$Vendor".ToLowerInvariant()) {
        '^intel$'    { return 'Intel' }
        '^amd$'      { return 'AMD' }
        '^qualcomm$' { return 'Qualcomm' }
        '^all$'      { return 'All' }
        '^auto$'     { return 'Auto' }
        default      { return 'Unknown' }
    }
}

# Detect the host's NPU vendor. On current AI-PC hardware the NPU brand tracks the
# CPU brand (Intel Core Ultra -> AI Boost, AMD Ryzen AI -> XDNA/IPU, Snapdragon X ->
# Hexagon), so CPU manufacturer is the reliable primary signal; a PnP probe enriches
# it for reporting and as a tie-breaker. Returns cpu/npu vendor + device string.
function Get-BenchmarkPlatform {
    $cpuName = ''; $cpuVendor = 'Unknown'
    try {
        $cpu = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop | Select-Object -First 1
        $cpuName = "$($cpu.Name)".Trim()
        switch -Regex ("$($cpu.Manufacturer)".Trim()) {
            'Intel'              { $cpuVendor = 'Intel' }
            'AMD|Advanced Micro' { $cpuVendor = 'AMD' }
            'Qualcomm'           { $cpuVendor = 'Qualcomm' }
        }
        if ($cpuVendor -eq 'Unknown') {
            switch -Regex ($cpuName) {
                'Intel'                      { $cpuVendor = 'Intel' }
                'AMD|Ryzen'                  { $cpuVendor = 'AMD' }
                'Snapdragon|Qualcomm|Oryon'  { $cpuVendor = 'Qualcomm' }
            }
        }
    } catch { }

    if ($cpuVendor -eq 'Unknown') {
        switch -Regex ("$env:PROCESSOR_IDENTIFIER") {
            'Intel'        { $cpuVendor = 'Intel' }
            'AMD'          { $cpuVendor = 'AMD' }
            'Qualcomm|ARM' { $cpuVendor = 'Qualcomm' }
        }
    }
    # The Ryzen AI SDK env var is an unambiguous AMD signal when CPU probing fails.
    if ($cpuVendor -eq 'Unknown' -and $env:RYZEN_AI_INSTALLATION_PATH) { $cpuVendor = 'AMD' }

    $npuDevice = ''; $npuPresent = $false
    try {
        $pnp = @(Get-PnpDevice -PresentOnly -ErrorAction Stop |
                Where-Object { $_.FriendlyName -match 'AI Boost|IPU|XDNA|NPU|Hexagon|Neural Proc' })
        if ($pnp.Count) { $npuPresent = $true; $npuDevice = $pnp[0].FriendlyName }
    } catch { }

    return [pscustomobject]@{
        cpu_vendor  = $cpuVendor
        cpu_name    = $cpuName
        npu_vendor  = $cpuVendor
        npu_device  = $npuDevice
        npu_present = $npuPresent
    }
}

# Should a variant with this backend run on a host with this NPU vendor? Neutral
# backends always run; vendor backends only when they match. If the host vendor is
# Unknown/All we can't (or shouldn't) filter, so everything is allowed.
function Test-BenchmarkVariantSupported {
    param(
        [Parameter(Mandatory)] [string]$Backend,
        [Parameter(Mandatory)] [string]$HostVendor
    )
    if ($HostVendor -eq 'All' -or $HostVendor -eq 'Unknown') { return $true }
    $vendor = Get-BenchmarkBackendVendor $Backend
    return ($vendor -eq 'Neutral' -or $vendor -eq $HostVendor)
}

# Resolve the effective host vendor to filter by, given a user selector
# ("auto" -> detect, "all" -> no filter, or an explicit vendor). Emits a banner and
# returns the canonical vendor name ('Intel'|'AMD'|'Qualcomm'|'All'|'Unknown').
function Resolve-BenchmarkHostVendor([string]$Selector, [object]$Platform = $null) {
    $sel = Get-BenchmarkVendorName $Selector
    if ($sel -eq 'All') {
        Write-Host "Platform filter: OFF (-NpuVendor all) -- attempting every variant." -ForegroundColor Cyan
        return 'All'
    }
    if ($sel -eq 'Intel' -or $sel -eq 'AMD' -or $sel -eq 'Qualcomm') {
        Write-Host ("Platform filter: forced NPU vendor = {0}." -f $sel) -ForegroundColor Cyan
        return $sel
    }
    # auto / unknown selector -> detect
    if (-not $Platform) { $Platform = Get-BenchmarkPlatform }
    $dev = if ($Platform.npu_device) { $Platform.npu_device } else { "no NPU device detected" }
    Write-Host ("Platform: {0} | NPU vendor {1} | {2}" -f `
            $Platform.cpu_name, $Platform.npu_vendor, $dev) -ForegroundColor Cyan
    if ($Platform.npu_vendor -eq 'Unknown') {
        Write-Host "  ! Could not determine NPU vendor; running all variants (use -NpuVendor to force)." -ForegroundColor Yellow
        return 'Unknown'
    }
    return $Platform.npu_vendor
}

# --- unified OVEP binary (Intel native ORT path) -----------------------------

# Path to the unified ONNX Runtime + OpenVINO EP binary. It ships openvino 2025.4.1
# DLLs that cannot share an output folder with the Intel GenAI build's openvino
# 2026.x, so it lives in its own "x64-ovep" tree (see msbuild\common.props). Intel
# NPUs are x64, so this path is x64-only by construction.
function Get-BenchmarkOvepExe([string]$Root, [string]$Configuration = "Release") {
    return (Join-Path $Root "build\x64-ovep\$Configuration\WhisperNpuHal.App.exe")
}

# Choose the exe to run a variant with. On an Intel host the neutral self-selecting
# variant ("auto"/onnx-static/ort backends) runs through the unified OVEP binary --
# the default Intel build is OpenVINO GenAI and has no ONNX Runtime backend compiled
# in, so it cannot serve the neutral static-ONNX path. Everything else (the OV-IR
# variants, and every non-Intel host) uses the default exe.
function Get-BenchmarkExeForVariant($Variant, [string]$DefaultExe, [string]$Root, [string]$HostVendor, [string]$Configuration = "Release") {
    if ($HostVendor -eq 'Intel' -and (Get-BenchmarkBackendVendor $Variant.backend) -eq 'Neutral') {
        $ovep = Get-BenchmarkOvepExe $Root $Configuration
        if (Test-Path $ovep) { return $ovep }
    }
    return $DefaultExe
}

# --- environment bootstrap ---------------------------------------------------

# Invoke a sibling setup/build script in an ISOLATED child PowerShell process.
# This is deliberate: get-eval-set.ps1 ends in `exit 0`, build.ps1/get-models.ps1
# `exit 1` on failure, and in PowerShell an `exit` inside a `& .\child.ps1` call
# terminates the PARENT script too -- which would silently kill the benchmark
# before the sweep. A child process contains the exit code so we can react to it.
function Invoke-BenchmarkChildScript {
    param(
        [Parameter(Mandatory)] [string]$ScriptPath,
        [string[]]$ScriptArgs = @(),
        [switch]$Fatal
    )
    $psExe = $null
    try { $psExe = (Get-Process -Id $PID -ErrorAction Stop).Path } catch { }
    if (-not $psExe) { $psExe = "powershell" }
    & $psExe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @ScriptArgs
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0 -and $Fatal) {
        throw ("{0} failed (exit {1})" -f [System.IO.Path]::GetFileName($ScriptPath), $code)
    }
    return $code
}

# Merge (idempotently) the neutral unified `auto` variant into a generated manifest.
# The unified binary self-selects its execution provider (NPU>GPU>CPU) at runtime, so
# this one entry runs the same self-selecting benchmark on any host. Runs on the
# portable static ONNX model. Returns $true if the manifest was written. No-op (returns
# $false) when the manifest or model is absent -- there is nothing to attach the entry to.
function Add-BenchmarkUnifiedAutoVariant {
    param(
        [Parameter(Mandatory)] [string]$Manifest,
        [Parameter(Mandatory)] [string]$Models
    )
    $modelRel = "models/whisper/en-static-onnx"
    $modelDir = Join-Path $Models "whisper\en-static-onnx"
    if (-not (Test-Path $Manifest) -or -not (Test-Path $modelDir)) { return $false }

    $sizeMb = [math]::Round(((Get-ChildItem $modelDir -Recurse -File -ErrorAction SilentlyContinue |
                Measure-Object Length -Sum).Sum / 1MB), 1)
    $entry = [ordered]@{
        id        = "whisper-tiny-unified-auto"
        backend   = "auto"
        precision = "fp32-static"
        method    = "unified ONNX Runtime; binary self-selects the EP (NPU>GPU>CPU)"
        model_dir = $modelRel
        devices   = @("auto")
        size_mb   = $sizeMb
    }
    $manifestObj = Get-Content $Manifest -Raw | ConvertFrom-Json
    $variants = @($manifestObj.variants | Where-Object { $_.id -ne $entry.id })
    $manifestObj.variants = @($variants + $entry)
    $manifestObj | ConvertTo-Json -Depth 8 | Set-Content -Path $Manifest -Encoding UTF8
    return $true
}

# Ensure everything a sweep needs exists, building the app and fetching
# models/eval/audio on demand so `benchmark-quant.ps1` works from a fresh checkout.
# Idempotent: every step is skipped when its output is already present. This
# assumes the toolchain + vendor SDK are installed (that is bootstrap.ps1's job);
# if the build can't run it says exactly that. Model fetches for the neutral
# onnx-static artifact require an authenticated `hf` CLI on PATH and are
# best-effort -- a missing private snapshot won't abort the run.
function Initialize-BenchmarkEnvironment {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [Parameter(Mandatory)] [string]$Exe,
        [Parameter(Mandatory)] [string]$Manifest,
        [Parameter(Mandatory)] [string]$EvalSet,
        [string]$Configuration = "Release",
        [object]$Platform = $null
    )
    if (-not $Platform) { $Platform = Get-BenchmarkPlatform }
    $vendor  = $Platform.npu_vendor
    $scripts = Join-Path $Root "scripts"
    $models  = Join-Path $Root "models"

    Write-Host ("== Bootstrap: preparing benchmark prerequisites (host {0}) ==" -f $vendor) -ForegroundColor Cyan

    # 1) Build the app for this host's backends if it is not built yet.
    if (-not (Test-Path $Exe)) {
        $buildArgs = @("-Configuration", $Configuration)
        switch ($vendor) {
            'AMD'      { $buildArgs += @("-EnableAmd", "-DisableIntel") }
            'Qualcomm' { $buildArgs += @("-EnableQualcomm", "-DisableIntel") }
            'Intel'    { }                       # Intel/OpenVINO is the default build
            default    { $buildArgs += @("-EnableOrt") }  # unknown host: at least the neutral ORT backend
        }
        Write-Host ("  - build app ({0})" -f ($buildArgs -join ' ')) -ForegroundColor DarkCyan
        Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "build.ps1") -ScriptArgs $buildArgs -Fatal | Out-Null
        if (-not (Test-Path $Exe)) {
            throw ("app build did not produce {0}. Install the toolchain/SDK first: .\scripts\bootstrap.ps1 -Platform {1}" -f $Exe, $vendor.ToLower())
        }
    } else {
        Write-Host "  - app already built" -ForegroundColor DarkGray
    }

    # 1b) Intel native path: also build the unified OVEP binary (ONNX Runtime +
    #     OpenVINO EP) so the self-selecting `auto` variant runs on the Intel
    #     NPU/GPU/CPU through ORT, mirroring AMD (VitisAI) and Qualcomm (QNN).
    #     Best-effort -- a failure here leaves the default Intel (GenAI) variants
    #     fully working; only the neutral ORT path is skipped.
    if ($vendor -eq 'Intel') {
        $ovepExe = Get-BenchmarkOvepExe $Root $Configuration
        if (-not (Test-Path $ovepExe)) {
            try {
                $ortLib = Join-Path $Root "third_party\onnxruntime-openvino\lib\onnxruntime.lib"
                if (-not (Test-Path $ortLib)) {
                    Write-Host "  - assemble ORT + OpenVINO EP distro (setup-ovep.ps1)" -ForegroundColor DarkCyan
                    Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "setup-ovep.ps1") | Out-Null
                }
                if (Test-Path $ortLib) {
                    Write-Host "  - build unified OVEP binary (build.ps1 -EnableOvep)" -ForegroundColor DarkCyan
                    Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "build.ps1") `
                        -ScriptArgs @("-Configuration", $Configuration, "-EnableOvep") | Out-Null
                }
            } catch {
                Write-Host ("  ! OVEP setup failed ({0}); Intel native ORT path will be skipped." -f $_.Exception.Message) -ForegroundColor Yellow
            }
            if (-not (Test-Path $ovepExe)) {
                Write-Host "  ! unified OVEP binary unavailable; the 'auto' variant will use the default exe on Intel." -ForegroundColor Yellow
            }
        } else {
            Write-Host "  - unified OVEP binary already built" -ForegroundColor DarkGray
        }
    }

    # 2) Sample audio (public) -- also the fallback clip for the eval set.
    if (-not (Test-Path (Join-Path $models "audio\jfk.wav"))) {
        Write-Host "  - fetch sample audio" -ForegroundColor DarkCyan
        Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-audio.ps1") | Out-Null
    }

    # 3) Models for this host's runnable variants. The private snapshot
    #    (get-models.ps1) carries the neutral onnx-static model + base manifest;
    #    the public per-vendor scripts fill in vendor models and merge their entry.
    $staticEnc = Join-Path $models "whisper\en-static-onnx\encoder_model.onnx"
    if (-not (Test-Path $Manifest) -or -not (Test-Path $staticEnc)) {
        if (Get-Command hf -ErrorAction SilentlyContinue) {
            Write-Host "  - fetch model snapshot (private HF; best-effort)" -ForegroundColor DarkCyan
            Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-models.ps1") | Out-Null
        } else {
            Write-Host "  ! neutral onnx-static model missing and 'hf' CLI not on PATH; skipping private snapshot." -ForegroundColor Yellow
        }
    }
    switch ($vendor) {
        'AMD' {
            if (-not (Test-Path (Join-Path $models "whisper\amd\tiny_encoder.onnx")) -or -not (Test-Path $Manifest)) {
                Write-Host "  - fetch AMD model + merge manifest" -ForegroundColor DarkCyan
                Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-amd-model.ps1") | Out-Null
            }
        }
        'Intel' {
            if (-not (Test-Path $Manifest)) {
                Write-Host "  - export Intel OpenVINO model" -ForegroundColor DarkCyan
                Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-model.ps1") | Out-Null
            }
        }
    }

    # 4) Eval set (public LibriSpeech, or jfk fallback).
    if (-not (Test-Path $EvalSet)) {
        Write-Host "  - build eval set" -ForegroundColor DarkCyan
        Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-eval-set.ps1") | Out-Null
    }

    # 5) Register the neutral self-selecting variant. The manifest is generated
    #    (gitignored), so this tracked step re-injects the unified `auto` entry on
    #    every run whenever the portable static ONNX model is present -- letting the
    #    same self-selecting benchmark run on any platform (x64 or ARM64).
    if (Test-Path $staticEnc) {
        if (Add-BenchmarkUnifiedAutoVariant -Manifest $Manifest -Models $models) {
            Write-Host "  - registered unified 'auto' variant (self-selecting EP)" -ForegroundColor DarkCyan
        }
    }

    # Final gate: these three are non-negotiable for a sweep.
    $missing = @($Manifest, $EvalSet, $Exe | Where-Object { -not (Test-Path $_) })
    if ($missing.Count) {
        throw ("bootstrap could not produce required prerequisites:`n  {0}`nRun .\scripts\bootstrap.ps1 -Platform {1} to install the toolchain/SDK and models." -f `
                ($missing -join "`n  "), $vendor.ToLower())
    }
    Write-Host "== Bootstrap: ready ==" -ForegroundColor Green
}

# --- detailed hardware inventory ---------------------------------------------

# Heuristic PCI (vendor:device) -> NPU architecture map. Windows reports only a
# generic device name ("NPU Compute Accelerator Device" on AMD), so the architecture
# is DERIVED from the PCI hardware id, not read from the device. Always cross-check
# against the raw pci_id the descriptor also carries; unknown ids return blank.
function Get-BenchmarkNpuArchitecture([string]$PciId) {
    switch ("$PciId".ToUpperInvariant()) {
        '1022:1502' { return 'AMD XDNA (Phoenix / Hawk Point)' }
        '1022:17F0' { return 'AMD XDNA 2 (Strix / Krackan Point)' }
        '1022:1640' { return 'AMD XDNA 2 (Strix Halo)' }
        '8086:7D1D' { return 'Intel AI Boost (Meteor / Arrow Lake NPU)' }
        '8086:643E' { return 'Intel AI Boost (Lunar Lake NPU 4)' }
        default { return '' }
    }
}

# Collect a detailed descriptor of the chips a benchmark can run on: CPU, every
# display adapter (integrated + discrete), and the NPU. All probes are best-effort
# (CIM/PnP can be absent or access-denied) and degrade to blank fields rather than
# throwing. AdapterRAM is a uint32 and saturates ~4 GB, so vram is approximate.
function Get-BenchmarkHardware {
    $cpu = [pscustomobject]@{
        name = ''; vendor = 'Unknown'; cores = $null; threads = $null
        max_clock_mhz = $null; description = ''
    }
    try {
        $c = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop | Select-Object -First 1
        $cpu.name = "$($c.Name)".Trim()
        $cpu.cores = [int]$c.NumberOfCores
        $cpu.threads = [int]$c.NumberOfLogicalProcessors
        $cpu.max_clock_mhz = [int]$c.MaxClockSpeed
        $cpu.description = "$($c.Description)".Trim()
        switch -Regex ("$($c.Manufacturer)") {
            'Intel'              { $cpu.vendor = 'Intel' }
            'AMD|Advanced Micro' { $cpu.vendor = 'AMD' }
            'Qualcomm'           { $cpu.vendor = 'Qualcomm' }
        }
    } catch { }

    $gpus = @()
    try {
        foreach ($g in Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop) {
            $vram = $null
            if ($g.AdapterRAM -and $g.AdapterRAM -gt 0) { $vram = [math]::Round($g.AdapterRAM / 1MB) }
            $res = $null
            if ($g.CurrentHorizontalResolution) { $res = "$($g.CurrentHorizontalResolution)x$($g.CurrentVerticalResolution)" }
            $gpus += [pscustomobject]@{
                name            = "$($g.Name)".Trim()
                video_processor = "$($g.VideoProcessor)".Trim()
                driver_version  = "$($g.DriverVersion)".Trim()
                vram_mb_approx  = $vram
                resolution      = $res
            }
        }
    } catch { }

    $npu = [pscustomobject]@{
        name = ''; architecture = ''; present = $false; manufacturer = ''
        pci_id = ''; driver_version = ''; instance_id = ''
    }
    try {
        $dev = @(Get-PnpDevice -PresentOnly -ErrorAction Stop |
                Where-Object { $_.FriendlyName -match 'AI Boost|IPU|XDNA|NPU|Hexagon|Neural Proc' })
        if ($dev.Count) {
            $d = $dev[0]
            $npu.present = $true
            $npu.name = "$($d.FriendlyName)".Trim()
            $npu.manufacturer = "$($d.Manufacturer)".Trim()
            $npu.instance_id = "$($d.InstanceId)".Trim()
            # The generic OS name has no architecture; the PCI vendor:device id does.
            if ("$($d.InstanceId)" -match 'VEN_([0-9A-Fa-f]{4})&DEV_([0-9A-Fa-f]{4})') {
                $npu.pci_id = ("{0}:{1}" -f $Matches[1], $Matches[2]).ToUpperInvariant()
                $npu.architecture = Get-BenchmarkNpuArchitecture $npu.pci_id
            }
            try {
                $dv = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction Stop).Data
                if ($dv) { $npu.driver_version = "$dv" }
            } catch { }
        }
    } catch { }

    $memGb = $null; $osName = ''
    try { $memGb = [math]::Round((Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory / 1GB, 1) } catch { }
    try { $osName = "$((Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).Caption)".Trim() } catch { }

    return [pscustomobject]@{
        collected_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        hostname      = $env:COMPUTERNAME
        os            = $osName
        memory_gb     = $memGb
        cpu           = $cpu
        gpus          = $gpus
        npu           = $npu
    }
}

# Human-readable markdown lines describing the collected hardware, for embedding in
# a report under a "## Host hardware" heading.
function Get-BenchmarkHardwareMarkdown([object]$Hardware) {
    $lines = New-Object System.Collections.Generic.List[string]
    $c = $Hardware.cpu
    $cpuBits = @()
    if ($c.cores) { $cpuBits += "$($c.cores)C" }
    if ($c.threads) { $cpuBits += "$($c.threads)T" }
    if ($c.max_clock_mhz) { $cpuBits += "$([math]::Round($c.max_clock_mhz/1000.0,2)) GHz" }
    $cpuSuffix = if ($cpuBits.Count) { " ($($cpuBits -join ', '))" } else { "" }
    $lines.Add("- **CPU**: ``$($c.name)`` [$($c.vendor)]$cpuSuffix")

    if ($Hardware.gpus -and $Hardware.gpus.Count) {
        foreach ($g in $Hardware.gpus) {
            $gBits = @()
            if ($g.driver_version) { $gBits += "driver $($g.driver_version)" }
            if ($g.vram_mb_approx) { $gBits += "~$($g.vram_mb_approx) MB" }
            $gSuffix = if ($gBits.Count) { " ($($gBits -join ', '))" } else { "" }
            $lines.Add("- **GPU**: ``$($g.name)``$gSuffix")
        }
    } else {
        $lines.Add("- **GPU**: (none detected)")
    }

    if ($Hardware.npu.present) {
        $n = $Hardware.npu
        # Lead with the derived architecture when known; otherwise the OS name. Always
        # keep the OS-reported name + PCI id visible so the derivation is verifiable.
        $headline = if ($n.architecture) { $n.architecture } else { $n.name }
        $nBits = @()
        if ($n.architecture -and $n.name) { $nBits += $n.name }
        elseif (-not $n.architecture -and $n.manufacturer) { $nBits += $n.manufacturer }
        if ($n.pci_id) { $nBits += "PCI $($n.pci_id)" }
        if ($n.driver_version) { $nBits += "driver $($n.driver_version)" }
        $nSuffix = if ($nBits.Count) { " ($($nBits -join ', '))" } else { "" }
        $lines.Add("- **NPU**: ``$headline``$nSuffix")
    } else {
        $lines.Add("- **NPU**: (none detected)")
    }

    $hostBits = @()
    if ($Hardware.memory_gb) { $hostBits += "$($Hardware.memory_gb) GB RAM" }
    if ($Hardware.os) { $hostBits += $Hardware.os }
    if ($hostBits.Count) { $lines.Add("- **Host**: $($hostBits -join ' | ')") }
    return $lines.ToArray()
}

# Persist the hardware descriptor as JSON next to a report for machine-readable use.
function Write-BenchmarkHostInfo([string]$Path, [object]$Hardware) {
    $parent = Split-Path $Path -Parent
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    ($Hardware | ConvertTo-Json -Depth 6) | Set-Content -Path $Path -Encoding UTF8
}

# Print a compact one-line-per-chip banner to the console.
function Write-BenchmarkHardwareBanner([object]$Hardware) {
    Write-Host "Host hardware:" -ForegroundColor Cyan
    foreach ($line in (Get-BenchmarkHardwareMarkdown $Hardware)) {
        Write-Host ("  " + ($line -replace '\*\*', '' -replace '`', '' -replace '^- ', ''))
    }
}

# --- app invocation ----------------------------------------------------------

# Run one clip through the app and return the parsed JSON record. Native STDERR
# (ORT/VitisAI warnings) is captured, not promoted to a terminating error. On
# failure returns [pscustomobject]@{ ok = $false; error = "..." }.
function Invoke-BenchmarkClip {
    param(
        [Parameter(Mandatory)] [string]$Exe,
        [Parameter(Mandatory)] [string]$ModelDir,
        [Parameter(Mandatory)] [string]$Audio,
        [Parameter(Mandatory)] [string]$Backend,
        [Parameter(Mandatory)] [string]$Device,
        [Parameter(Mandatory)] [int]$Runs,
        [Parameter(Mandatory)] [string]$CacheDir,
        [Parameter(Mandatory)] [string]$Ref,
        [int]$Threads = 0,
        [string]$Provider = "",
        [switch]$HotOnly
    )
    $a = @($ModelDir, $Audio, $Backend, $Device, "$Runs", "--cache", $CacheDir, "--ref", $Ref, "--json")
    if ($Threads -gt 0) { $a += @("--threads", "$Threads") }
    if ($Provider) { $a += @("--provider", $Provider) }
    if ($HotOnly) { $a += "--hot-only" }

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & $Exe @a 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldEap
    }
    $line = ($raw | Where-Object { $_ -match '^\{"ok"' } | Select-Object -Last 1)
    if (-not $line) { return [pscustomobject]@{ ok = $false; error = "no-json-output (exit $exitCode)" } }
    return $line | ConvertFrom-Json
}

# --- aggregation -------------------------------------------------------------

# Cold/hot engine-load seconds come from the first successful clip's record
# (load happens once per engine, not per clip). Tolerates the legacy load_warm_s
# field name for hot start.
function Get-BenchmarkLoadTimes($FirstRow) {
    $hot = if ($FirstRow.PSObject.Properties.Name -contains "load_hot_s") { $FirstRow.load_hot_s } else { $FirstRow.load_warm_s }
    return [pscustomobject]@{ cold = $FirstRow.load_cold_s; hot = $hot }
}

# Aggregate a set of successful clip records into the means/sums both harnesses
# report. WER/CER are micro-averaged (sum edits / sum refs), not averaged per clip.
function Measure-BenchmarkClips($Rows) {
    $wEdits = ($Rows | Measure-Object word_edits -Sum).Sum
    $wRef = ($Rows | Measure-Object ref_words -Sum).Sum
    $cEdits = ($Rows | Measure-Object char_edits -Sum).Sum
    $cRef = ($Rows | Measure-Object ref_chars -Sum).Sum
    return [pscustomobject]@{
        count         = $Rows.Count
        mean_ms       = ($Rows | Measure-Object mean_ms -Average).Average
        mean_rtf      = ($Rows | Measure-Object rtf -Average).Average
        mean_tps      = ($Rows | Measure-Object throughput_tps -Average).Average
        mean_logprob  = ($Rows | Measure-Object avg_logprob -Average).Average
        mean_ttft_ms  = ($Rows | Measure-Object ttft_ms -Average).Average
        mean_tpot_ms  = ($Rows | Measure-Object tpot_ms -Average).Average
        audio_seconds = ($Rows | Measure-Object audio_len_s -Sum).Sum
        word_edits    = $wEdits
        ref_words     = $wRef
        char_edits    = $cEdits
        ref_chars     = $cRef
        wer           = if ($wRef) { $wEdits / $wRef } else { $null }
        cer           = if ($cRef) { $cEdits / $cRef } else { $null }
        first_row     = ($Rows | Select-Object -First 1)
        last_row      = ($Rows | Select-Object -Last 1)
    }
}
