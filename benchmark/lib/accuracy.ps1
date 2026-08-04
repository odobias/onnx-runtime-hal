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

function Test-BenchmarkNpuOperationAssignment {
    param([object]$Result, [string]$RequestedDevice)
    if ($RequestedDevice -ne "npu") { return $true }
    if (-not $Result) { return $false }
    $effective = if ($Result.first_row) { $Result.first_row } else { $Result }
    foreach ($field in @("assigned_ops_cpu", "assigned_ops_npu", "operation_assignment_source")) {
        if (-not ($effective.PSObject.Properties.Name -contains $field)) { return $false }
    }
    try {
        return [int]$effective.assigned_ops_cpu -ge 0 -and
            [int]$effective.assigned_ops_npu -gt 0 -and
            -not [string]::IsNullOrWhiteSpace([string]$effective.operation_assignment_source)
    } catch {
        return $false
    }
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
        basis = "fixture-expected-pred"
    }
}

function Normalize-BenchmarkTranscript([string]$Text) {
    if ($null -eq $Text) { return "" }
    $normalized = $Text.ToUpperInvariant()
    $normalized = [regex]::Replace($normalized, "[^A-Z0-9']+", " ")
    return $normalized.Trim()
}

# Resolve the baked baseline for one eval row. Workloads that declare a
# `baselineKey` in the manifest are graded strictly against their own entry under
# `baselines`, so a package with different weights (for example multilingual tiny)
# is never compared to another package's frozen hypotheses. Workloads without a
# key keep using the flat baseline_* fields.
function Get-BenchmarkClipBaseline {
    param(
        [Parameter(Mandatory)][AllowNull()][object]$EvalRow,
        [string]$BaselineKey = ""
    )
    if ($null -eq $EvalRow) { return $null }
    if ($BaselineKey) {
        $map = $EvalRow.baselines
        if (-not $map -or -not ($map.PSObject.Properties.Name -contains $BaselineKey)) {
            return $null
        }
        $entry = $map.$BaselineKey
        if (-not $entry -or -not ($entry.PSObject.Properties.Name -contains "wer")) { return $null }
        return [pscustomobject]@{
            hyp = [string]$entry.hyp
            wer = [double]$entry.wer
            cer = [double]$entry.cer
            lang = [string]$entry.lang
            task = [string]$entry.task
        }
    }
    if (-not ($EvalRow.PSObject.Properties.Name -contains "baseline_wer")) { return $null }
    return [pscustomobject]@{
        hyp = [string]$EvalRow.baseline_hyp
        wer = [double]$EvalRow.baseline_wer
        cer = [double]$EvalRow.baseline_cer
        lang = [string]$EvalRow.baseline_lang
        task = [string]$EvalRow.baseline_task
    }
}

function Measure-BenchmarkWhisperBaselineAgreement {
    param(
        [Parameter(Mandatory)][object[]]$ClipResults,
        [Parameter(Mandatory)][object[]]$EvalRows,
        [Parameter(Mandatory)][object]$Aggregate,
        [string]$BaselineKey = ""
    )
    $byId = @{}
    foreach ($row in $EvalRows) { $byId[[string]$row.id] = $row }

    $finite = (Test-BenchmarkFiniteNumber $Aggregate.wer) -and
        (Test-BenchmarkFiniteNumber $Aggregate.cer)
    $compared = 0
    $hypMismatches = 0
    $langDrifts = 0
    $taskDrifts = 0
    $baselineWordEdits = 0.0
    $baselineRefWords = 0.0
    $baselineCharEdits = 0.0
    $baselineRefChars = 0.0
    $clipDeltas = [System.Collections.Generic.List[object]]::new()

    foreach ($clip in $ClipResults) {
        $id = if ($clip.eval_id) { [string]$clip.eval_id } else { [string]$clip.id }
        $baseline = Get-BenchmarkClipBaseline -EvalRow $byId[$id] -BaselineKey $BaselineKey
        if (-not $baseline) { continue }
        $compared++
        $hyp = if ($clip.text) { [string]$clip.text } else { [string]$clip.transcription }
        $baselineHyp = $baseline.hyp
        if ($baselineHyp -and
            (Normalize-BenchmarkTranscript $hyp) -ne (Normalize-BenchmarkTranscript $baselineHyp)) {
            $hypMismatches++
        }
        # Whisper picks the language per clip, so a run that detects a different one
        # is transcribing (or worse, translating) something else entirely -- treat it
        # as drift even when WER happens to land nearby. Baselines predating language
        # capture record none, and stay uncompared.
        $lang = [string]$clip.language
        $baselineLang = $baseline.lang
        $langMatches = (-not $baselineLang) -or ($lang -eq $baselineLang)
        if (-not $langMatches) { $langDrifts++ }
        # Same reasoning for transcribe vs translate: the transcript is answering a
        # different question, so it must not be graded as though it were the baseline's.
        $task = [string]$clip.task
        $baselineTask = $baseline.task
        $taskMatches = (-not $baselineTask) -or ($task -eq $baselineTask)
        if (-not $taskMatches) { $taskDrifts++ }
        $baselineWer = $baseline.wer
        $baselineCer = $baseline.cer
        $clipWer = [double]$clip.wer
        $clipCer = [double]$clip.cer
        if (Test-BenchmarkFiniteNumber $clip.ref_words) {
            $baselineRefWords += [double]$clip.ref_words
            $baselineWordEdits += $baselineWer * [double]$clip.ref_words
        }
        if (Test-BenchmarkFiniteNumber $clip.ref_chars) {
            $baselineRefChars += [double]$clip.ref_chars
            $baselineCharEdits += $baselineCer * [double]$clip.ref_chars
        }
        $clipDeltas.Add([ordered]@{
            id = $id
            wer = $clipWer
            baseline_wer = $baselineWer
            wer_delta = $clipWer - $baselineWer
            cer = $clipCer
            baseline_cer = $baselineCer
            cer_delta = $clipCer - $baselineCer
            hyp_matches_baseline = (
                -not $baselineHyp -or
                (Normalize-BenchmarkTranscript $hyp) -eq (Normalize-BenchmarkTranscript $baselineHyp)
            )
            language = $lang
            baseline_language = $baselineLang
            language_matches_baseline = $langMatches
            task = $task
            baseline_task = $baselineTask
            task_matches_baseline = $taskMatches
        })
    }

    $baselineWerAgg = if ($baselineRefWords -gt 0) {
        $baselineWordEdits / $baselineRefWords
    } else { $null }
    $baselineCerAgg = if ($baselineRefChars -gt 0) {
        $baselineCharEdits / $baselineRefChars
    } else { $null }

    return [pscustomobject]@{
        outputs_finite = $finite
        compared = $compared
        decision_flips = $hypMismatches
        language_drifts = $langDrifts
        task_drifts = $taskDrifts
        max_abs_probability_difference = $null
        basis = $(if ($compared -gt 0) {
            "human-transcript-reference+baked-baseline"
        } else {
            "human-transcript-reference"
        })
        baseline_wer = $baselineWerAgg
        baseline_cer = $baselineCerAgg
        wer_delta = if ($null -ne $baselineWerAgg -and (Test-BenchmarkFiniteNumber $Aggregate.wer)) {
            [double]$Aggregate.wer - [double]$baselineWerAgg
        } else { $null }
        cer_delta = if ($null -ne $baselineCerAgg -and (Test-BenchmarkFiniteNumber $Aggregate.cer)) {
            [double]$Aggregate.cer - [double]$baselineCerAgg
        } else { $null }
        clips = @($clipDeltas)
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
        [Parameter(Mandatory)][string]$Root,
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
    $npuOperationAssignmentRecorded =
        Test-BenchmarkNpuOperationAssignment $Result $RequestedDevice
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
            npu_operation_assignment_recorded = $npuOperationAssignmentRecorded
            baseline_compared = ([int]$ReferenceAgreement.compared -gt 0)
            no_language_drift = ([int]$ReferenceAgreement.language_drifts -eq 0)
            no_task_drift = ([int]$ReferenceAgreement.task_drifts -eq 0)
            valid = $Valid
        }
        metrics = ConvertTo-BenchmarkPublishedValue -Root $Root -Value $Result
    }
}
