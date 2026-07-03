# Shared benchmark harness library. This is the single home for the concerns that
# every benchmark entry point (benchmark.ps1, compare-devices.ps1, run.ps1) needs:
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
    return @(
        "timestamp_utc", "requested_backend", "resolved_backend", "device", "device_name", "device_full_name",
        "model_dir", "audio_path", "audio_seconds", "runs", "warmup", "cache_dir",
        "cold_load_seconds", "warm_load_seconds", "mean_infer_seconds", "rtf", "realtime_factor",
        "label", "model_size_mb", "avg_logprob", "ttft_ms", "tpot_ms", "throughput_tps",
        "wer", "cer", "transcription",
        "runtime", "model_format", "decode_strategy", "max_context", "eval_clips", "status",
        "cold_start_seconds", "hot_start_seconds", "power_source"
    )
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

# --- detailed hardware inventory ---------------------------------------------

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

    $npu = [pscustomobject]@{ name = ''; present = $false; manufacturer = ''; driver_version = ''; instance_id = '' }
    try {
        $dev = @(Get-PnpDevice -PresentOnly -ErrorAction Stop |
                Where-Object { $_.FriendlyName -match 'AI Boost|IPU|XDNA|NPU|Hexagon|Neural Proc' })
        if ($dev.Count) {
            $d = $dev[0]
            $npu.present = $true
            $npu.name = "$($d.FriendlyName)".Trim()
            $npu.manufacturer = "$($d.Manufacturer)".Trim()
            $npu.instance_id = "$($d.InstanceId)".Trim()
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
        $nBits = @()
        if ($Hardware.npu.manufacturer) { $nBits += $Hardware.npu.manufacturer }
        if ($Hardware.npu.driver_version) { $nBits += "driver $($Hardware.npu.driver_version)" }
        $nSuffix = if ($nBits.Count) { " ($($nBits -join ', '))" } else { "" }
        $lines.Add("- **NPU**: ``$($Hardware.npu.name)``$nSuffix")
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
        [string]$Provider = ""
    )
    $a = @($ModelDir, $Audio, $Backend, $Device, "$Runs", "--cache", $CacheDir, "--ref", $Ref, "--json")
    if ($Threads -gt 0) { $a += @("--threads", "$Threads") }
    if ($Provider) { $a += @("--provider", $Provider) }

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
