# Validates the accuracy evidence for the current environment: the suite writes one
# results/accuracy-runs/<environment-snapshot-id>.jsonl per environment, and the last
# invocation in the attempt ledger says which of them is current.
#
# -All validates every ledger ever written instead, which is a different and much
# stronger claim -- that no run on any past build or host ever failed the gate. One
# honestly-recorded failure (an NPU run that fell back to the GPU, say) then fails this
# check forever, and the only ways out are to keep -All out of CI or to delete evidence
# of a real failure. Hence the default.
#
#   .\benchmark\validate-accuracy.ps1
#   .\benchmark\validate-accuracy.ps1 -All
#   .\benchmark\validate-accuracy.ps1 -Ledger results\accuracy-runs\<id>.jsonl

[CmdletBinding()]
param(
    [string]$Ledger = "",
    [switch]$All,
    [switch]$AllowInvalid
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\harness.ps1")
Initialize-BenchmarkConsole
$root = Split-Path $PSScriptRoot -Parent
$isAggregateValidation = -not [bool]$Ledger
$runDirectory = Join-Path $root "results\accuracy-runs"
$scope = "explicit ledger"
$ledgerPaths = if ($Ledger) {
    @($Ledger)
} elseif ($All) {
    $scope = "every recorded environment"
    $paths = @()
    $legacyLedger = Join-Path $root "results\ledgers\accuracy.jsonl"
    if (Test-Path -LiteralPath $legacyLedger) { $paths += $legacyLedger }
    if (Test-Path -LiteralPath $runDirectory) {
        $paths += @(Get-ChildItem -LiteralPath $runDirectory -Filter "*.jsonl" -File |
            Sort-Object Name | Select-Object -ExpandProperty FullName)
    }
    $paths
} else {
    # Which environment is current comes from the last invocation in the rolling attempt
    # ledger, not from file timestamps: restoring or copying an old evidence file would
    # otherwise make it look like the newest run and quietly change what gets validated.
    $attemptLedger = Join-Path $root "results\ledgers\attempts.jsonl"
    $currentId = ""
    if (Test-Path -LiteralPath $attemptLedger) {
        $attempts = @(Get-Content -LiteralPath $attemptLedger -Encoding UTF8 |
            Where-Object { $_.Trim() })
        if ($attempts.Count) {
            $currentId = [string](($attempts[-1] | ConvertFrom-Json).environment_snapshot_id)
        }
    }
    if (-not $currentId) {
        throw ("Cannot tell which environment is current: $attemptLedger is missing or empty. " +
            "Run benchmark\run-suite.ps1, or pass -Ledger <file> / -All.")
    }
    $currentPath = Join-Path $runDirectory "$currentId.jsonl"
    if (-not (Test-Path -LiteralPath $currentPath)) {
        throw "The last suite invocation ($currentId) recorded no accuracy evidence: $currentPath"
    }
    $scope = "current environment $currentId"
    @($currentPath)
}
foreach ($path in $ledgerPaths) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Accuracy ledger does not exist: $path" }
}
$records = @($ledgerPaths | ForEach-Object {
    Get-Content -LiteralPath $_ -Encoding UTF8 |
        Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
})
if (-not $records.Count) { throw "Accuracy evidence is empty." }

$errors = [Collections.Generic.List[string]]::new()
foreach ($record in $records) {
    foreach ($field in @(
        "measurement_purpose", "workload_id", "execution_profile", "graph_role",
        "runtime_target", "requested_device", "resolved_provider", "model_sha256",
        "fixture_sha256", "graph_artifacts", "environment_snapshot_id",
        "model_compilation_provenance_ids", "reference_agreement", "trust_gate", "metrics"
    )) {
        if (-not ($record.PSObject.Properties.Name -contains $field)) {
            $errors.Add("$($record.workload_id)/$($record.execution_profile): missing $field")
        }
    }
    if ($record.measurement_purpose -ne "accuracy-quick") {
        $errors.Add("$($record.workload_id): unexpected purpose '$($record.measurement_purpose)'")
    }
    if (-not $record.reference_agreement.outputs_finite) {
        $errors.Add("$($record.workload_id)/$($record.execution_profile): non-finite output")
    }
    if ([int]$record.reference_agreement.decision_flips -ne 0) {
        $errors.Add("$($record.workload_id)/$($record.execution_profile): reference decision flip")
    }
    if ([int]$record.reference_agreement.language_drifts -ne 0) {
        $errors.Add("$($record.workload_id)/$($record.execution_profile): detected language differs from baseline")
    }
    if ([int]$record.reference_agreement.task_drifts -ne 0) {
        $errors.Add("$($record.workload_id)/$($record.execution_profile): Whisper task differs from baseline")
    }
    $intended = Test-BenchmarkIntendedProvider $record.metrics ([string]$record.requested_device)
    if ([bool]$record.trust_gate.intended_provider_resolved -ne $intended) {
        $errors.Add(
            "$($record.workload_id)/$($record.execution_profile): recorded intended-provider result " +
            "does not match provider evidence")
    }
    if ($record.trust_gate.valid -and -not $intended) {
        $errors.Add("$($record.workload_id)/$($record.execution_profile): valid record used fallback provider")
    }
    if ($record.trust_gate.PSObject.Properties.Name -contains "npu_operation_assignment_recorded") {
        $assignmentRecorded =
            Test-BenchmarkNpuOperationAssignment $record.metrics ([string]$record.requested_device)
        if ([bool]$record.trust_gate.npu_operation_assignment_recorded -ne $assignmentRecorded) {
            $errors.Add(
                "$($record.workload_id)/$($record.execution_profile): recorded NPU operation-assignment " +
                "gate does not match assignment evidence")
        }
        if ($record.requested_device -eq "npu" -and $record.trust_gate.valid -and
            -not $assignmentRecorded) {
            $errors.Add(
                "$($record.workload_id)/$($record.execution_profile): valid NPU record lacks " +
                "CPU/NPU operation assignment")
        }
    }
    if (-not $AllowInvalid -and -not $record.trust_gate.valid) {
        $errors.Add("$($record.workload_id)/$($record.execution_profile): trust gate invalid")
    }
    $snapshotPath = Join-Path $root "results\host-snapshots\$($record.environment_snapshot_id).json"
    if (-not (Test-Path -LiteralPath $snapshotPath)) {
        $errors.Add("$($record.workload_id): missing environment snapshot $($record.environment_snapshot_id)")
    }
    foreach ($provenanceId in @($record.model_compilation_provenance_ids)) {
        $provenancePath = Join-Path $root "results\model-compilation\$provenanceId.json"
        if (-not (Test-Path -LiteralPath $provenancePath)) {
            $errors.Add("$($record.workload_id): missing compilation provenance $provenanceId")
        }
    }
}

$wholeNpu = @($records | Where-Object {
    $_.workload_id -eq "fakeaudio" -and $_.requested_device -eq "npu" -and $_.graph_role -eq "whole"
})
if ($wholeNpu.Count) { $errors.Add("whole-graph FakeAudio NPU appeared in the accuracy ledger") }

$whisperCohorts = @($records | Where-Object { $_.workload_kind -eq "asr" } |
    Select-Object -ExpandProperty execution_profile -Unique)
# Only demanded of a scope that recorded ASR at all: a classifier-only run has no
# Whisper cohorts to be missing.
if ($isAggregateValidation -and $whisperCohorts.Count -and
    ("static-onnx" -notin $whisperCohorts -or "dynamic-kv-onnx" -notin $whisperCohorts)) {
    $errors.Add("static and dynamic Whisper cohorts were not both recorded")
}

if ($errors.Count) {
    # -ErrorAction Continue: the script-wide "Stop" preference would otherwise make
    # the first Write-Error terminate and hide every remaining finding.
    $errors | ForEach-Object { Write-Error $_ -ErrorAction Continue }
    exit 1
}

$valid = @($records | Where-Object { $_.trust_gate.valid }).Count
$npuAssignments = @($records | Where-Object {
    $_.requested_device -eq "npu" -and
    (Test-BenchmarkNpuOperationAssignment $_.metrics ([string]$_.requested_device))
}).Count
$runtimes = @($records | Select-Object -ExpandProperty runtime_target -Unique | Sort-Object)
Write-Host "Accuracy evidence valid: $valid / $($records.Count) records" -ForegroundColor Green
Write-Host "Scope                : $scope"
Write-Host "Accuracy sources     : $($ledgerPaths.Count)"
Write-Host "Runtime targets      : $($runtimes -join ', ')"
Write-Host "Whisper cohorts      : $($whisperCohorts -join ', ')"
Write-Host "Whole FakeAudio NPU  : absent" -ForegroundColor Green
Write-Host "NPU op assignments   : $npuAssignments records"
