# Runner-package manifest loading and hardware-aware pack selection.

function ConvertTo-BenchmarkArchitecture([object]$Architecture) {
    $value = [string]$Architecture
    if ($value -match '(?i)arm64') { return 'ARM64' }
    if ($value -match '(?i)x64|amd64') { return 'x64' }
    throw "Unsupported runner architecture '$value'."
}

function Get-BenchmarkHostArchitecture {
    return (ConvertTo-BenchmarkArchitecture `
        ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture))
}

function Get-BenchmarkRunnerPackageManifest {
    param([Parameter(Mandatory)][string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $manifest = Get-Content -LiteralPath $resolved -Raw -Encoding UTF8 |
        ConvertFrom-Json -ErrorAction Stop
    if ([int]$manifest.schema_version -ne 1) {
        throw "Unsupported runner package schema '$($manifest.schema_version)' in $resolved."
    }
    if (-not $manifest.architecture -or -not $manifest.runners) {
        throw "Runner package manifest is missing architecture or runners: $resolved"
    }
    $ids = @{}
    foreach ($runner in @($manifest.runners)) {
        if (-not $runner.id -or -not $runner.path -or -not $runner.executable) {
            throw "Runner package manifest has an incomplete runner entry: $resolved"
        }
        $runnerPath = ([string]$runner.path).Replace('\', '/')
        if ($runnerPath -notmatch '^runners/[^/]+$' -or $runnerPath -match '(^|/)\.\.(/|$)') {
            throw "Runner package manifest contains an unsafe path '$runnerPath'."
        }
        if ((ConvertTo-BenchmarkArchitecture $runner.architecture) -ne
            (ConvertTo-BenchmarkArchitecture $manifest.architecture)) {
            throw "Runner '$($runner.id)' architecture differs from its package."
        }
        if ($ids.ContainsKey([string]$runner.id)) {
            throw "Runner package manifest contains duplicate runner '$($runner.id)'."
        }
        $ids[[string]$runner.id] = $true
    }
    return [pscustomobject]@{
        path = $resolved
        root = Split-Path $resolved -Parent
        value = $manifest
    }
}

function Test-BenchmarkRunnerProvider {
    param([object]$Runner, [string]$Provider)
    $token = if ($Provider) { $Provider } else { "auto" }
    foreach ($pattern in @($Runner.provider_patterns)) {
        if ($token -match [string]$pattern) { return $true }
    }
    return $false
}

function Resolve-BenchmarkRunnerPack {
    param(
        [Parameter(Mandatory)][object]$PackageManifest,
        [Parameter(Mandatory)][ValidateSet("bundled", "winml")][string]$Runtime,
        [string]$Provider = "",
        [Parameter(Mandatory)][string]$HostVendor,
        [string]$HostArchitecture = "",
        [int]$OsBuild = 0
    )

    if (-not $HostArchitecture) { $HostArchitecture = Get-BenchmarkHostArchitecture }
    $HostArchitecture = ConvertTo-BenchmarkArchitecture $HostArchitecture
    $packageArchitecture = ConvertTo-BenchmarkArchitecture $PackageManifest.value.architecture
    if ($packageArchitecture -ne $HostArchitecture) {
        throw "Runner package architecture '$packageArchitecture' cannot run on '$HostArchitecture'."
    }
    if ($OsBuild -le 0) {
        try { $OsBuild = [Environment]::OSVersion.Version.Build } catch { $OsBuild = 0 }
    }

    $rejected = [System.Collections.Generic.List[string]]::new()
    $candidates = @()
    foreach ($runner in @($PackageManifest.value.runners)) {
        if ([string]$runner.runtime_target -ne $Runtime) { continue }
        if (-not (Test-BenchmarkRunnerProvider -Runner $runner -Provider $Provider)) { continue }

        $vendors = @($runner.vendors | ForEach-Object { [string]$_ })
        $vendorMatch = $vendors -contains "Any" -or $vendors -contains $HostVendor
        if (-not $vendorMatch) {
            $rejected.Add("$($runner.id): host vendor '$HostVendor' is unsupported")
            continue
        }
        if ($runner.minimum_os_build -and $OsBuild -lt [int]$runner.minimum_os_build) {
            $rejected.Add("$($runner.id): requires Windows build $($runner.minimum_os_build)")
            continue
        }
        $runnerRoot = Join-Path $PackageManifest.root ([string]$runner.path -replace '/', '\')
        $exe = Join-Path $runnerRoot ([string]$runner.executable)
        if (-not (Test-Path -LiteralPath $exe)) {
            $rejected.Add("$($runner.id): executable is missing")
            continue
        }
        $candidates += [pscustomobject]@{
            id = [string]$runner.id
            runtime_target = [string]$runner.runtime_target
            platform_tag = [string]$runner.platform_tag
            root = $runnerRoot
            exe = $exe
            priority = [int]$runner.priority
            manifest_entry = $runner
        }
    }
    $selected = @($candidates | Sort-Object priority -Descending | Select-Object -First 1)
    if ($selected.Count) { return $selected[0] }

    $providerLabel = if ($Provider) { $Provider } else { "auto" }
    $details = if ($rejected.Count) { " " + ($rejected -join "; ") } else { "" }
    throw "No packaged runner supports runtime '$Runtime', provider '$providerLabel', vendor '$HostVendor', architecture '$HostArchitecture'.$details"
}

function Get-BenchmarkPackagedRuntimeSelections {
    param(
        [Parameter(Mandatory)][object]$PackageManifest,
        [string]$Provider = "",
        [Parameter(Mandatory)][string]$HostVendor,
        [string]$HostArchitecture = "",
        [int]$OsBuild = 0
    )
    $resolved = [System.Collections.Generic.List[object]]::new()
    foreach ($runtime in @("bundled", "winml")) {
        try {
            $resolved.Add((Resolve-BenchmarkRunnerPack -PackageManifest $PackageManifest `
                -Runtime $runtime -Provider $Provider -HostVendor $HostVendor `
                -HostArchitecture $HostArchitecture -OsBuild $OsBuild))
        } catch {
            if ($runtime -eq "bundled") { throw }
        }
    }
    return @($resolved)
}
