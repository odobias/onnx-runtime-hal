function Get-BenchmarkExecutionProfiles {
    param(
        [Parameter(Mandatory)][object]$Workload,
        [Parameter(Mandatory)][string]$Device,
        [Parameter(Mandatory)][string[]]$Selectors,
        [Parameter(Mandatory)][string]$HostVendor
    )
    $explicitSelection = $Selectors -notcontains "all"
    $profiles = @($Workload.executionProfiles)
    if (-not $profiles.Count) {
        $profiles = @([pscustomobject]@{
            id = "default"; devices = @($Workload.devices); graphRole = "whole"
            accuracyEligible = $true
        })
    }
    return @($profiles | Where-Object {
        $profile = $_
        $selected = -not $explicitSelection -or $Selectors -contains [string]$profile.id
        $defaultEligible = $explicitSelection -or
            -not ($profile.PSObject.Properties.Name -contains "default") -or
            [bool]$profile.default
        $deviceEligible = @($profile.devices) -contains $Device
        $vendorEligible = -not $profile.hostVendors -or @($profile.hostVendors) -contains $HostVendor
        $selected -and $defaultEligible -and $deviceEligible -and $vendorEligible
    })
}

function Get-BenchmarkEvalRows {
    param(
        [Parameter(Mandatory)][string]$Root,
        [string]$Audio = ""
    )
    $manifest = Join-Path $Root "src\workloads\eval\eval.jsonl"
    if (-not (Test-Path -LiteralPath $manifest)) { return @() }
    $rows = @(Get-Content -LiteralPath $manifest -Encoding UTF8 |
        Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
    if (-not $Audio) { return $rows }
    $id = [IO.Path]::GetFileNameWithoutExtension($Audio)
    return @($rows | Where-Object { $_.id -eq $id })
}

function Ensure-BenchmarkClassifierFixtures {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][hashtable]$FixtureDirectories,
        [Parameter(Mandatory)][string[]]$Models,
        [switch]$Regenerate
    )
    $need = @($Models | Where-Object {
        $Regenerate -or -not (Test-Path -LiteralPath (Join-Path $FixtureDirectories[$_] "model.tsv"))
    })
    if (-not $need.Count) { return }
    $python = Join-Path $Root "artifacts/venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host "Fixtures missing for $($need -join ', '); Python generation is unavailable." -ForegroundColor Yellow
        return
    }
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    & $python (Join-Path $Root "tools\fixtures\generate.py") --models @need
    if ($LASTEXITCODE -ne 0) { throw "Classifier fixture generation failed with exit $LASTEXITCODE." }
}

function Write-BenchmarkAttemptRecord {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$MeasurementPurpose,
        [Parameter(Mandatory)][object]$Workload,
        [Parameter(Mandatory)][object]$ExecutionProfile,
        [Parameter(Mandatory)][string]$RequestedDevice,
        [Parameter(Mandatory)][string]$RequestedProvider,
        [Parameter(Mandatory)][string]$Status,
        [string]$ErrorMessage = "",
        [object]$Result = $null,
        [Parameter(Mandatory)][string]$EnvironmentSnapshotId,
        [string[]]$CompilationProvenanceIds = @()
    )
    $record = [ordered]@{
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        suite = "portable"
        measurement_purpose = $MeasurementPurpose
        workload = [string]$Workload.id
        kind = [string]$Workload.kind
        backend = [string]$Workload.backend
        execution_profile = [string]$ExecutionProfile.id
        graph_role = [string]$ExecutionProfile.graphRole
        requested_device = $RequestedDevice
        requested_provider = $RequestedProvider
        status = $Status
        error = $ErrorMessage
        environment_snapshot_id = $EnvironmentSnapshotId
        model_compilation_provenance_ids = $CompilationProvenanceIds
        result = $Result
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Path -Parent) | Out-Null
    Add-Content -LiteralPath $Path -Value ($record | ConvertTo-Json -Depth 20 -Compress) -Encoding UTF8
}

function Write-BenchmarkRunAttemptSnapshot {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$RollingLedger,
        [Parameter(Mandatory)][string]$EnvironmentSnapshotId
    )
    if (-not (Test-Path -LiteralPath $RollingLedger)) {
        throw "Rolling attempt ledger does not exist: $RollingLedger"
    }

    $matches = [Collections.Generic.List[string]]::new()
    $lineNumber = 0
    foreach ($rawLine in [IO.File]::ReadLines($RollingLedger)) {
        ++$lineNumber
        $line = $rawLine.TrimStart([char]0xFEFF)
        if (-not $line.Trim()) { continue }
        try {
            $record = $line | ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Rolling attempt ledger contains invalid JSON at line ${lineNumber}: $($_.Exception.Message)"
        }
        if ([string]$record.environment_snapshot_id -eq $EnvironmentSnapshotId) {
            $matches.Add($line)
        }
    }
    if (-not $matches.Count) {
        throw "No attempt records found for environment snapshot '$EnvironmentSnapshotId'."
    }

    $dir = Join-Path $Root "results\run-attempts"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $path = Join-Path $dir "$EnvironmentSnapshotId.jsonl"
    $content = ($matches -join "`n") + "`n"
    $encoding = [Text.UTF8Encoding]::new($false)
    if (Test-Path -LiteralPath $path) {
        $existing = [IO.File]::ReadAllText($path, $encoding).Replace("`r`n", "`n")
        if ($existing -ne $content) {
            throw "Published attempt snapshot already exists with different content: $path"
        }
    } else {
        [IO.File]::WriteAllText($path, $content, $encoding)
    }
    return [pscustomobject]@{ path = $path; count = $matches.Count }
}
