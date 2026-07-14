[CmdletBinding()]
param(
    [string]$Ledger = "",
    [switch]$AllowInvalid
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\harness.ps1")
Initialize-BenchmarkConsole
$root = Split-Path $PSScriptRoot -Parent
if (-not $Ledger) { $Ledger = Join-Path $root "results\ledgers\accuracy.jsonl" }
if (-not (Test-Path -LiteralPath $Ledger)) { throw "Accuracy ledger does not exist: $Ledger" }

$records = @(Get-Content -LiteralPath $Ledger -Encoding UTF8 |
    Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if (-not $records.Count) { throw "Accuracy ledger is empty: $Ledger" }

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
if ("static-onnx" -notin $whisperCohorts -or "dynamic-kv-onnx" -notin $whisperCohorts) {
    $errors.Add("static and dynamic Whisper cohorts were not both recorded")
}

if ($errors.Count) {
    $errors | ForEach-Object { Write-Error $_ }
    exit 1
}

$valid = @($records | Where-Object { $_.trust_gate.valid }).Count
$runtimes = @($records | Select-Object -ExpandProperty runtime_target -Unique | Sort-Object)
Write-Host "Accuracy ledger valid: $valid / $($records.Count) records" -ForegroundColor Green
Write-Host "Runtime targets      : $($runtimes -join ', ')"
Write-Host "Whisper cohorts      : $($whisperCohorts -join ', ')"
Write-Host "Whole FakeAudio NPU  : absent" -ForegroundColor Green
