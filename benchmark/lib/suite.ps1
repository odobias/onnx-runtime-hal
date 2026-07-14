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
    $manifest = Join-Path $Root "workloads\eval\eval.jsonl"
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
    $python = Join-Path $Root ".venv\Scripts\python.exe"
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
