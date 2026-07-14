# Benchmark library module. Dot-source benchmark/lib/harness.ps1 instead of loading this directly.

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
                driver_date     = $(try { ([datetime]$g.DriverDate).ToString("o") } catch { "" })
                pnp_id          = "$($g.PNPDeviceID)".Trim()
                vram_mb_approx  = $vram
                resolution      = $res
            }
        }
    } catch { }

    $npu = [pscustomobject]@{
        name = ''; architecture = ''; present = $false; manufacturer = ''
        pci_id = ''; driver_version = ''; driver_date = ''; instance_id = ''
    }
    try {
        $dev = @(Get-PnpDevice -PresentOnly -ErrorAction Stop |
                Where-Object { $_.FriendlyName -match 'AI Boost|\bIPU\b|XDNA|\bNPU\b|Hexagon|Neural Proc' })
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
            try {
                $dd = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_DriverDate' -ErrorAction Stop).Data
                if ($dd) { $npu.driver_date = ([datetime]$dd).ToString("o") }
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
