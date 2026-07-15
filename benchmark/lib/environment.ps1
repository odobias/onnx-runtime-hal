# Benchmark library module. Dot-source benchmark/lib/harness.ps1 instead of loading this directly.

function Get-BenchmarkPathHash([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    $item = Get-Item -LiteralPath $Path
    if (-not $item.PSIsContainer) {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $lines = Get-ChildItem -LiteralPath $Path -File -Recurse |
            Sort-Object FullName |
            ForEach-Object {
                $relative = $_.FullName.Substring($item.FullName.Length).TrimStart('\', '/').Replace('\', '/')
                "$relative`t$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())`n"
            }
        $bytes = [Text.Encoding]::UTF8.GetBytes(($lines -join ""))
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-BenchmarkObjectId([object]$Value) {
    $json = $Value | ConvertTo-Json -Depth 20 -Compress
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($json)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant().Substring(0, 20)
    } finally {
        $sha.Dispose()
    }
}

function Get-BenchmarkEnvironmentSnapshot {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string]$RuntimeTarget,
        [Parameter(Mandatory)][string]$BuildTree
    )
    $hardware = Get-BenchmarkHardware
    $os = $null; $bios = $null; $system = $null
    try { $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop } catch {}
    try { $bios = Get-CimInstance Win32_BIOS -ErrorAction Stop } catch {}
    try { $system = Get-CimInstance Win32_ComputerSystemProduct -ErrorAction Stop } catch {}
    $biosDate = $null
    try { if ($bios.ReleaseDate) { $biosDate = ([datetime]$bios.ReleaseDate).ToString("o") } } catch {}
    $ubr = $null
    try { $ubr = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction Stop).UBR } catch {}
    $powerPlan = ""
    try { $powerPlan = ((powercfg /getactivescheme 2>$null) -join " ").Trim() } catch {}
    $energySaver = $null
    try {
        $energySaver = [bool](Get-ItemPropertyValue `
            'HKCU:\Software\Microsoft\Windows\CurrentVersion\BackgroundAccessApplications' `
            -Name GlobalUserDisabled -ErrorAction Stop)
    } catch {}
    $powerSource = "unknown"
    $batteryDetails = $null
    try {
        $battery = Get-CimInstance Win32_Battery -ErrorAction Stop | Select-Object -First 1
        if (-not $battery) { $powerSource = "AC" }
        elseif ($battery.BatteryStatus -in @(2, 6, 7, 8, 9, 11)) { $powerSource = "AC" }
        else { $powerSource = "battery" }
        if ($battery) {
            $batteryDetails = [ordered]@{
                device_id = "$($battery.DeviceID)"
                status = "$($battery.Status)"
                battery_status = $battery.BatteryStatus
                estimated_charge_remaining_pct = $battery.EstimatedChargeRemaining
                estimated_runtime_minutes = $battery.EstimatedRunTime
                design_voltage_mv = $battery.DesignVoltage
                chemistry = $battery.Chemistry
            }
        }
    } catch {}
    $powerOverlays = [ordered]@{ ac = ""; dc = "" }
    try {
        $powerKey = Get-ItemProperty `
            'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes' `
            -ErrorAction Stop
        $powerOverlays.ac = "$($powerKey.ActiveOverlayAcPowerScheme)"
        $powerOverlays.dc = "$($powerKey.ActiveOverlayDcPowerScheme)"
    } catch {}
    $thermalZones = @()
    try {
        foreach ($zone in @(Get-CimInstance -Namespace root/wmi `
                -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop)) {
            $thermalZones += [ordered]@{
                instance_name = "$($zone.InstanceName)"
                temperature_c = [math]::Round(($zone.CurrentTemperature / 10.0) - 273.15, 2)
            }
        }
    } catch {}

    $binaries = @()
    $exeDir = Split-Path $Exe -Parent
    foreach ($file in @(Get-ChildItem -LiteralPath $exeDir -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in @('.dll', '.exe') } | Sort-Object Name)) {
        $vi = $file.VersionInfo
        $binaries += [ordered]@{
            name = $file.Name
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            file_version = "$($vi.FileVersion)"
            product_version = "$($vi.ProductVersion)"
        }
    }
    $sdkManifests = @()
    $thirdParty = Join-Path $Root "third_party"
    foreach ($manifest in @(Get-ChildItem -LiteralPath $thirdParty -Filter "VERSION.json" `
            -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName)) {
        try {
            $sdkManifests += [ordered]@{
                path = $manifest.FullName.Substring($Root.Length).TrimStart('\').Replace('\', '/')
                sha256 = (Get-FileHash -LiteralPath $manifest.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                metadata = (Get-Content -LiteralPath $manifest.FullName -Raw -Encoding UTF8 | ConvertFrom-Json)
            }
        } catch {}
    }

    $snapshot = [ordered]@{
        schema_version = 1
        collected_utc = [DateTime]::UtcNow.ToString("o")
        runtime_target = $RuntimeTarget
        build_tree = $BuildTree
        executable = [ordered]@{
            path = $Exe
            sha256 = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        host = [ordered]@{
            hostname = $env:COMPUTERNAME
            architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
            os_caption = "$($os.Caption)"
            os_version = "$($os.Version)"
            windows_build = "$($os.BuildNumber)"
            ubr = $ubr
            memory_bytes = $os.TotalVisibleMemorySize * 1KB
            system_product = "$($system.Name)"
            system_vendor = "$($system.Vendor)"
            bios_vendor = "$($bios.Manufacturer)"
            bios_version = "$($bios.SMBIOSBIOSVersion)"
            bios_release_date = $biosDate
        }
        hardware = $hardware
        power = [ordered]@{
            source = $powerSource
            active_plan = $powerPlan
            energy_saver_enabled = $energySaver
            overlay_scheme = $powerOverlays
            battery = $batteryDetails
            thermal_zones = $thermalZones
        }
        runtime_binaries = $binaries
        vendor_sdk_manifests = $sdkManifests
    }
    $id = Get-BenchmarkObjectId $snapshot
    $snapshot["id"] = $id
    $dir = Join-Path $Root "results\host-snapshots"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $path = Join-Path $dir "$id.json"
    if (-not (Test-Path -LiteralPath $path)) {
        ($snapshot | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $path -Encoding UTF8
    }
    return [pscustomobject]@{ id = $id; path = $path; value = $snapshot }
}

function Write-BenchmarkCompilationProvenance {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$ArtifactPath,
        [Parameter(Mandatory)][string]$ProfileId,
        [Parameter(Mandatory)][string]$GraphRole,
        [Parameter(Mandatory)][string]$HostVendor,
        [Parameter(Mandatory)][object]$EnvironmentSnapshot
    )
    $artifact = $ArtifactPath
    if (Test-Path -LiteralPath $ArtifactPath -PathType Container) {
        $modelTsv = Join-Path $ArtifactPath "model.tsv"
        if (Test-Path -LiteralPath $modelTsv) {
            $first = (Get-Content -LiteralPath $modelTsv -Encoding UTF8 | Select-Object -First 1)
            if ($first) {
                $candidate = Join-Path $ArtifactPath (($first -split "`t")[0])
                $artifact = [IO.Path]::GetFullPath($candidate)
            }
        }
    }
    $providerBinaryPattern = switch ($HostVendor) {
        "Qualcomm" { "QnnHtp.dll" }
        "Intel" { "openvino.dll" }
        "AMD" { "onnxruntime_vitisai_ep.dll" }
        default { "" }
    }
    $providerBinary = @($EnvironmentSnapshot.value.runtime_binaries |
        Where-Object { $providerBinaryPattern -and $_.name -ieq $providerBinaryPattern } |
        Select-Object -First 1)
    $sdkEvidence = @($EnvironmentSnapshot.value.vendor_sdk_manifests |
        Where-Object { $_.metadata.vendor -eq $HostVendor } | Select-Object -First 1)
    $version = "unknown"
    $versionSource = "unknown: no authoritative SDK manifest or provider binary version was found"
    $compilerName = $(if ($providerBinaryPattern) { "$HostVendor execution-provider compiler" } else { "unknown" })
    if ($sdkEvidence.Count) {
        $metadata = $sdkEvidence[0].metadata
        $compilerName = if ($metadata.package) { [string]$metadata.package } else { $compilerName }
        $candidateVersion = @(
            $metadata.version, $metadata.qnn_ep_package_version, $metadata.openvino_version
        ) | Where-Object { $_ } | Select-Object -First 1
        if ($candidateVersion) {
            $version = [string]$candidateVersion
            $versionSource = "SDK package metadata: $($sdkEvidence[0].path)"
        }
    } elseif ($providerBinary.Count -and $providerBinary[0].file_version) {
        $version = "$($providerBinary[0].file_version)"
        $versionSource = "runtime binary file version: $($providerBinary[0].name)"
    }
    $step = $null
    $stepPath = Join-Path $ArtifactPath "fixture-step.json"
    if (Test-Path -LiteralPath $stepPath) {
        try { $step = Get-Content -LiteralPath $stepPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch {}
    }
    $record = [ordered]@{
        schema_version = 1
        created_utc = [DateTime]::UtcNow.ToString("o")
        provenance_scope = "pre-execution-input-artifact-and-compiler-context"
        artifact_stage = "provider-input"
        execution_profile = $ProfileId
        graph_role = $GraphRole
        target_vendor = $HostVendor
        target_architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        artifact_path = $artifact
        artifact_sha256 = Get-BenchmarkPathHash $artifact
        source_model_sha256 = $(if ($step -and $step.source_model_sha256) { "$($step.source_model_sha256)" } else { "unknown" })
        compiler = [ordered]@{
            name = $compilerName
            version = $version
            version_source = $versionSource
        }
        compile_or_export_options = $(if ($step) { $step } else { [ordered]@{ status = "unknown"; reason = "no generated fixture-step metadata" } })
        transformation_recipe = $(if ($step -and $step.transform) { "$($step.transform)" } else { "none-recorded" })
        compiled_cache_artifact = [ordered]@{
            status = "not-captured"
            reason = "provider compilation occurs during session creation after this input-artifact record is generated"
        }
        compilation_host_snapshot_id = $EnvironmentSnapshot.id
    }
    $id = Get-BenchmarkObjectId $record
    $record["id"] = $id
    $dir = Join-Path $Root "results\model-compilation"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $path = Join-Path $dir "$id.json"
    if (-not (Test-Path -LiteralPath $path)) {
        ($record | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $path -Encoding UTF8
    }
    return [pscustomobject]@{ id = $id; path = $path; value = $record }
}

# Print a compact one-line-per-chip banner to the console.
