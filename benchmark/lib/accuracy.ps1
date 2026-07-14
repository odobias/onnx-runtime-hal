function Test-BenchmarkFiniteNumber($Value) {
    if ($null -eq $Value) { return $false }
    try {
        $number = [double]$Value
        return -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)
    } catch {
        return $false
    }
}

function Test-BenchmarkIntendedProvider {
    param([object]$Result, [string]$RequestedDevice)
    if (-not $Result) { return $false }
    $effective = if ($Result.first_row) { $Result.first_row } else { $Result }
    if ($effective.fallback_occurred -eq $true) { return $false }
    $resolved = if ($effective.resolved_provider) {
        [string]$effective.resolved_provider
    } else {
        [string]$effective.execution_provider
    }
    if (-not $resolved) { return $false }
    if ($RequestedDevice -ne "cpu" -and $resolved -match "(?i)^CPUExecutionProvider$") {
        return $false
    }
    if ($RequestedDevice -eq "npu" -and $resolved -match "(?i)dml|directml") {
        return $false
    }
    $requestedProvider = [string]$effective.requested_provider
    if ($RequestedDevice -eq "npu" -and $requestedProvider -match "(?i)prefer_(gpu|cpu)") {
        return $false
    }
    foreach ($attempt in @($effective.provider_attempts)) {
        if (-not $attempt.success) { continue }
        $provider = [string]$attempt.provider
        if ($RequestedDevice -eq "npu" -and $provider -match "(?i)prefer_(gpu|cpu)") {
            return $false
        }
        if ($RequestedDevice -eq "gpu" -and $provider -match "(?i)prefer_cpu") {
            return $false
        }
        break
    }
    return $true
}

function Measure-BenchmarkClassifierAgreement {
    param([Parameter(Mandatory)][object]$Result)
    $finite = $true
    $flips = 0
    $compared = 0
    foreach ($sample in @($Result.samples)) {
        if (-not (Test-BenchmarkFiniteNumber $sample.p)) { $finite = $false }
        if ($sample.expected_pred) {
            $compared++
            if ([string]$sample.pred -ne [string]$sample.expected_pred) { $flips++ }
        }
    }
    return [pscustomobject]@{
        outputs_finite = $finite
        compared = $compared
        decision_flips = $flips
        max_abs_probability_difference = $Result.max_abs_p_diff
    }
}

function Write-BenchmarkAccuracyRecord {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][object]$Record
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $Path -Parent) | Out-Null
    Add-Content -LiteralPath $Path -Value ($Record | ConvertTo-Json -Depth 30 -Compress) -Encoding UTF8
}

function New-BenchmarkAccuracyRecord {
    param(
        [Parameter(Mandatory)][object]$Workload,
        [Parameter(Mandatory)][object]$ExecutionProfile,
        [Parameter(Mandatory)][string]$RuntimeTarget,
        [Parameter(Mandatory)][string]$RequestedDevice,
        [Parameter(Mandatory)][string]$RequestedProvider,
        [Parameter(Mandatory)][string]$ModelHash,
        [Parameter(Mandatory)][string]$FixtureHash,
        [Parameter(Mandatory)][object]$GraphArtifacts,
        [Parameter(Mandatory)][string]$EnvironmentSnapshotId,
        [Parameter(Mandatory)][string[]]$CompilationProvenanceIds,
        [Parameter(Mandatory)][object]$Result,
        [Parameter(Mandatory)][object]$ReferenceAgreement,
        [Parameter(Mandatory)][bool]$Valid,
        [Parameter(Mandatory)][bool]$IntendedProviderResolved
    )
    $first = if ($Result.first_row) { $Result.first_row } else { $Result }
    return [ordered]@{
        schema_version = 1
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        measurement_purpose = "accuracy-quick"
        workload_id = [string]$Workload.id
        workload_kind = [string]$Workload.kind
        execution_profile = [string]$ExecutionProfile.id
        graph_role = [string]$ExecutionProfile.graphRole
        frontend_contract = [string]$ExecutionProfile.frontendContract
        reference_contract = [string]$ExecutionProfile.referenceContract
        runtime_target = $RuntimeTarget
        requested_device = $RequestedDevice
        requested_provider = $RequestedProvider
        resolved_provider = $(if ($first.resolved_provider) { [string]$first.resolved_provider } else { [string]$first.execution_provider })
        fallback_occurred = [bool]$first.fallback_occurred
        model_sha256 = $ModelHash
        fixture_sha256 = $FixtureHash
        graph_artifacts = $GraphArtifacts
        environment_snapshot_id = $EnvironmentSnapshotId
        model_compilation_provenance_ids = $CompilationProvenanceIds
        reference_agreement = $ReferenceAgreement
        trust_gate = [ordered]@{
            outputs_finite = [bool]$ReferenceAgreement.outputs_finite
            no_reference_decision_flips = ([int]$ReferenceAgreement.decision_flips -eq 0)
            intended_provider_resolved = $IntendedProviderResolved
            accuracy_eligible = -not ($ExecutionProfile.PSObject.Properties.Name -contains "accuracyEligible") -or [bool]$ExecutionProfile.accuracyEligible
            valid = $Valid
        }
        metrics = $Result
    }
}
