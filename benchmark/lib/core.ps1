# Benchmark library module. Dot-source benchmark/lib/harness.ps1 instead of loading this directly.

# Shared benchmark harness library. This is the single home for the concerns that
# every benchmark entry point (benchmark-onnx.ps1, benchmark-quant.ps1, compare-devices.ps1, run.ps1) needs:
# the canonical results-CSV schema, the app invocation + JSON parse, clip aggregation
# math, and best-effort power-source detection.
#
# Dot-source it from a sibling script:
#     . (Join-Path $PSScriptRoot "benchmark/lib/harness.ps1")
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

# --- publication privacy ----------------------------------------------------

function Get-BenchmarkAnonymousId {
    param(
        [Parameter(Mandatory)][string]$Namespace,
        [Parameter(Mandatory)][string]$Value
    )
    if (-not $Value) { return "" }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes("${Namespace}`0${Value}")
        $digest = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
        return $digest.Substring(0, 16)
    } finally {
        $sha.Dispose()
    }
}

function Get-BenchmarkHostId {
    $machineIdentity = $env:COMPUTERNAME
    try {
        $machineGuid = Get-ItemPropertyValue `
            'HKLM:\SOFTWARE\Microsoft\Cryptography' -Name MachineGuid -ErrorAction Stop
        if ($machineGuid) { $machineIdentity = "$machineGuid|$machineIdentity" }
    } catch {}
    return "host-" + (Get-BenchmarkAnonymousId -Namespace "benchmark-host" -Value $machineIdentity)
}

function Get-BenchmarkPublishRoot([Parameter(Mandatory)][string]$Root) {
    return [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
}

function ConvertTo-BenchmarkPublishedPath {
    param(
        [Parameter(Mandatory)][string]$Root,
        [AllowEmptyString()][string]$Path
    )
    if (-not $Path) { return "" }
    if ($Path.StartsWith('${REPO_ROOT}', [StringComparison]::Ordinal) -or
        $Path.StartsWith('external:', [StringComparison]::Ordinal)) {
        return $Path
    }
    try {
        $rootPath = Get-BenchmarkPublishRoot $Root
        $fullPath = [IO.Path]::GetFullPath($Path)
        if ($fullPath.Equals($rootPath, [StringComparison]::OrdinalIgnoreCase)) {
            return '${REPO_ROOT}'
        }
        $rootPrefix = $rootPath + [IO.Path]::DirectorySeparatorChar
        if ($fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            $relative = $fullPath.Substring($rootPrefix.Length).Replace('\', '/')
            return '${REPO_ROOT}/' + $relative
        }
        $id = Get-BenchmarkAnonymousId -Namespace "external-path" -Value $fullPath
        return "external:$id"
    } catch {
        return "external:" + (Get-BenchmarkAnonymousId -Namespace "invalid-path" -Value $Path)
    }
}

function ConvertTo-BenchmarkPublishedValue {
    param(
        [Parameter(Mandatory)][string]$Root,
        [AllowNull()]$Value,
        [string]$PropertyName = ""
    )
    if ($null -eq $Value) { return $null }

    $pathProperties = @(
        "artifact_path", "audio", "audio_path", "cache_dir", "executable_path",
        "model_dir", "model_path", "path", "working_directory"
    )
    if ($Value -is [string]) {
        if ($PropertyName -in $pathProperties) {
            return ConvertTo-BenchmarkPublishedPath -Root $Root -Path $Value
        }
        return $Value.Replace(
            (Get-BenchmarkPublishRoot $Root), '${REPO_ROOT}', [StringComparison]::OrdinalIgnoreCase)
    }
    if ($Value -is [Collections.IDictionary]) {
        $result = [ordered]@{}
        foreach ($key in $Value.Keys) {
            $result[$key] = ConvertTo-BenchmarkPublishedValue `
                -Root $Root -Value $Value[$key] -PropertyName ([string]$key)
        }
        return $result
    }
    if ($Value -is [Collections.IEnumerable] -and $Value -isnot [string]) {
        return @($Value | ForEach-Object {
            ConvertTo-BenchmarkPublishedValue -Root $Root -Value $_ -PropertyName $PropertyName
        })
    }
    if ($Value -is [Management.Automation.PSCustomObject]) {
        $result = [ordered]@{}
        foreach ($property in $Value.PSObject.Properties) {
            $result[$property.Name] = ConvertTo-BenchmarkPublishedValue `
                -Root $Root -Value $property.Value -PropertyName $property.Name
        }
        return $result
    }
    return $Value
}

function Get-BenchmarkPublishedPowerPlan([string]$PowerPlan) {
    if ($PowerPlan -match '(?i)[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}') {
        return $Matches[0].ToLowerInvariant()
    }
    if ($PowerPlan) { return "custom" }
    return ""
}

# --- results CSV schema (single source of truth) -----------------------------

# Canonical column order for results/ledgers/asr.csv. This is the authoritative
# list for the PowerShell harness; the C++ app (app/main.cpp) writes the same schema
# for its single-run rows and MUST be kept in sync with this list. See results/README.md.
function Get-BenchmarkResultColumns {
    # AUTHORITATIVE column order. Must stay byte-identical to kBenchmarkCsvHeader in
    # app/main.cpp (the other writer of this ledger). Change both + results/README.md.
    return @(
        "timestamp_utc", "requested_backend", "resolved_backend", "device", "device_name", "device_full_name",
        "model_package", "model_sha256", "variant_id", "base_model", "precision", "quant_method", "execution_provider",
        "model_dir", "audio_path", "audio_seconds", "runs", "warmup", "cache_dir",
        "cold_load_seconds", "hot_load_seconds", "warm_load_seconds", "mean_infer_seconds", "rtf", "realtime_factor",
        "label", "model_size_mb", "avg_logprob", "ttft_ms", "tpot_ms", "throughput_tps",
        "wer", "cer", "transcription",
        "runtime", "model_format", "decode_strategy", "max_context", "eval_clips", "status",
        "cold_start_seconds", "hot_start_seconds", "power_source",
        "host_arch", "host_os", "runtime_version", "inference_precision",
        "requested_provider", "resolved_provider", "fallback_occurred", "provider_attempts",
        "ep_nodes", "cpu_nodes", "cpu_offload_pct", "cpu_offload_ops",
        "assigned_ops_cpu", "assigned_ops_npu", "operation_assignment_source",
        "measurement_purpose", "execution_profile", "graph_role",
        "environment_snapshot_id", "model_compilation_provenance_ids", "error"
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
                $oldLine = ($columns | ForEach-Object {
                    $value = Get-BenchmarkOptional $old $_ ""
                    if ($_ -eq "hot_load_seconds" -and -not $value) {
                        $value = Get-BenchmarkOptional $old "hot_start_seconds" `
                            (Get-BenchmarkOptional $old "warm_load_seconds" "")
                    }
                    ConvertTo-BenchmarkCsvCell $value
                }) -join ","
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
                Where-Object { $_.FriendlyName -match 'AI Boost|\bIPU\b|XDNA|\bNPU\b|Hexagon|Neural Proc' })
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
# 2026.x, so it lives in its own "x64-ovep" tree (see eng\msbuild\common.props). Intel
# NPUs are x64, so this path is x64-only by construction.
