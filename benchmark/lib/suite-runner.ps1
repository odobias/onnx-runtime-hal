# Profile execution orchestration. Loaded by benchmark/lib/harness.ps1.
function Write-SuiteAttempt {
    param(
        [object]$Workload, [object]$ExecutionProfile, [string]$RequestedDevice,
        [string]$Status, [string]$ErrorMessage, [object]$Result,
        [string[]]$CompilationProvenanceIds = @()
    )
    Write-BenchmarkAttemptRecord -Path $AttemptLedger -MeasurementPurpose $Mode `
        -Workload $Workload -ExecutionProfile $ExecutionProfile `
        -RequestedDevice $RequestedDevice `
        -RequestedProvider $(if ($Provider) { $Provider } else { "auto" }) `
        -Status $Status -ErrorMessage $ErrorMessage -Result $Result `
        -EnvironmentSnapshotId $environmentSnapshot.id `
        -CompilationProvenanceIds $CompilationProvenanceIds
}

function Invoke-NativeBenchmark {
    param([object]$Workload, [object]$ExecutionProfile, [string]$RequestedDevice)

    $relPath = if ($ExecutionProfile.path) { [string]$ExecutionProfile.path } else { [string]$Workload.path }
    if ($ExecutionProfile.outputs) {
        $needsBuild = [bool]$RegenerateFixtures
        foreach ($output in @($ExecutionProfile.outputs)) {
            if (-not (Test-Path (Join-Path $root (([string]$output) -replace '/', '\')))) {
                $needsBuild = $true
            }
        }
        if ($needsBuild) {
            $vpy = Join-Path $root ".venv\Scripts\python.exe"
            if (-not (Test-Path $vpy)) {
                $message = "fixture recipe requires the project .venv: $($ExecutionProfile.path)"
                Write-Host $message -ForegroundColor Yellow
                Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "assets-missing" $message $null
                return $false
            }
            foreach ($step in @($ExecutionProfile.steps)) {
                $script = Join-Path $root (([string]$step.script) -replace '/', '\')
                $stepArgs = @($step.arguments | ForEach-Object { [string]$_ })
                Write-Host "fixture step: $($step.script) $($stepArgs -join ' ')" -ForegroundColor DarkGray
                & $vpy $script @stepArgs
                if ($LASTEXITCODE -ne 0) {
                    $message = "fixture step failed ($LASTEXITCODE): $($step.script)"
                    Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "fixture-failed" $message $null
                    return $false
                }
            }
            foreach ($output in @($ExecutionProfile.outputs)) {
                $outputPath = Join-Path $root (([string]$output) -replace '/', '\')
                if (-not (Test-Path $outputPath)) {
                    $message = "fixture recipe did not produce: $outputPath"
                    Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "fixture-failed" $message $null
                    return $false
                }
            }
        }
        Write-Host "fixture    : $($ExecutionProfile.id)/$RequestedDevice -> $relPath" -ForegroundColor DarkGray
    }

    $path = Join-Path $root ($relPath -replace '/', '\')
    if (-not (Test-Path $path)) {
        $message = "workload assets missing: $path"
        Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "assets-missing" $message $null
        return $false
    }

    $provenanceIds = @()
    $graphArtifacts = [ordered]@{}
    if ($ExecutionProfile.frontendContract) {
        $frontend = Join-Path $root "models\deepfake\fakeaudio\model.frontend-fp32.onnx"
        if (Test-Path -LiteralPath $frontend) {
            $frontendProvenance = Write-BenchmarkCompilationProvenance `
                -Root $root -ArtifactPath $frontend -ProfileId ([string]$ExecutionProfile.id) `
                -GraphRole "frontend" -HostVendor "CPU" -EnvironmentSnapshot $environmentSnapshot
            $provenanceIds += $frontendProvenance.id
            $graphArtifacts["frontend_sha256"] = $frontendProvenance.value.artifact_sha256
        }
    }
    $modelProvenance = Write-BenchmarkCompilationProvenance `
        -Root $root -ArtifactPath $path -ProfileId ([string]$ExecutionProfile.id) `
        -GraphRole ([string]$ExecutionProfile.graphRole) -HostVendor $hostVendor `
        -EnvironmentSnapshot $environmentSnapshot
    $provenanceIds += $modelProvenance.id
    $artifactKey = if ($ExecutionProfile.graphRole -eq "backbone") { "backbone_sha256" } else { "whole_sha256" }
    $graphArtifacts[$artifactKey] = $modelProvenance.value.artifact_sha256

    $env:NPU_INFERENCE_BENCH_MEASUREMENT_PURPOSE = $Mode
    $env:NPU_INFERENCE_BENCH_EXECUTION_PROFILE = [string]$ExecutionProfile.id
    $env:NPU_INFERENCE_BENCH_GRAPH_ROLE = [string]$ExecutionProfile.graphRole
    $env:NPU_INFERENCE_BENCH_ENVIRONMENT_SNAPSHOT_ID = $environmentSnapshot.id
    $env:NPU_INFERENCE_BENCH_MODEL_PROVENANCE_IDS = $provenanceIds -join ";"

    $cache = Get-BenchmarkCachePath -Root $root -Platform $platform -Provider $providerTag `
        -Workload ([string]$Workload.id) -Profile ([string]$ExecutionProfile.id) `
        -Device $RequestedDevice
    Reset-BenchmarkCache -Path $cache | Out-Null

    if ($Mode -eq "accuracy-quick" -and $Workload.kind -eq "asr") {
        $evalRows = Get-BenchmarkEvalRows -Root $root -Audio $(if ($audioExplicit) { $Audio } else { "" })
        if (-not $evalRows.Count) {
            $message = "no Whisper evaluation rows were found"
            Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "fixtures-missing" $message $null $provenanceIds
            return $false
        }
        $clipResults = @()
        for ($i = 0; $i -lt $evalRows.Count; $i++) {
            $eval = $evalRows[$i]
            $clipPath = Join-Path $root (([string]$eval.audio -replace '/', '\'))
            if (-not (Test-Path -LiteralPath $clipPath)) {
                $clipPath = Join-Path $root "workloads\eval\$($eval.id).wav"
            }
            $invoke = @{
                Exe = $exe; ModelDir = $path; Audio = $clipPath
                Backend = [string]$Workload.backend; Device = $RequestedDevice
                Runs = 1; CacheDir = $cache; Ref = [string]$eval.ref; Provider = $Provider
            }
            if ($i -gt 0) { $invoke.HotOnly = $true }
            $clip = Invoke-BenchmarkClip @invoke
            if (-not $clip.ok) {
                $message = "Whisper evaluation clip '$($eval.id)' failed: $($clip.error)"
                Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "executor-failed" $message $clip $provenanceIds
                return $false
            }
            $clip | Add-Member -NotePropertyName eval_id -NotePropertyValue ([string]$eval.id)
            $clipResults += $clip
        }
        $aggregate = Measure-BenchmarkClips $clipResults
        $aggregate | Add-Member -NotePropertyName clips -NotePropertyValue $clipResults
        $finite = (Test-BenchmarkFiniteNumber $aggregate.wer) -and
            (Test-BenchmarkFiniteNumber $aggregate.cer)
        $intended = @($clipResults | Where-Object {
            -not (Test-BenchmarkIntendedProvider $_ $RequestedDevice)
        }).Count -eq 0
        $agreement = [pscustomobject]@{
            outputs_finite = $finite
            compared = $clipResults.Count
            decision_flips = 0
            basis = "human-transcript-reference"
        }
        $eligible = -not ($ExecutionProfile.PSObject.Properties.Name -contains "accuracyEligible") -or
            [bool]$ExecutionProfile.accuracyEligible
        $valid = $finite -and $intended -and $eligible
        $record = New-BenchmarkAccuracyRecord -Workload $Workload `
            -ExecutionProfile $ExecutionProfile -RuntimeTarget $Runtime `
            -RequestedDevice $RequestedDevice `
            -RequestedProvider $(if ($Provider) { $Provider } else { "auto" }) `
            -ModelHash (Get-BenchmarkPathHash $path) `
            -FixtureHash (Get-BenchmarkPathHash (Join-Path $root "workloads\eval\eval.jsonl")) `
            -GraphArtifacts $graphArtifacts `
            -EnvironmentSnapshotId $environmentSnapshot.id `
            -CompilationProvenanceIds $provenanceIds -Result $aggregate `
            -ReferenceAgreement $agreement -Valid $valid -IntendedProviderResolved $intended
        Write-BenchmarkAccuracyRecord -Path $AccuracyLedger -Record $record
        Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice `
            $(if ($valid) { "accuracy-valid" } else { "accuracy-invalid" }) "" $record $provenanceIds
        return $valid
    }

    $args = if ($Workload.kind -eq "asr") {
        @("run", "whisper", $path, $Audio, [string]$Workload.backend, $RequestedDevice, "$Runs",
            "--cache", $cache, "--json")
    } else {
        @("run", [string]$Workload.id, $path, $RequestedDevice, "$ClassifierRuns",
            "--cache", $cache, "--json")
    }
    if ($Provider) { $args += @("--provider", $Provider) }
    if ($Mode -eq "latency" -and -not $NoResults) {
        $ledger = if ($Workload.kind -eq "asr") { $Results } else { $ClassifierResults }
        $args += @("--results", $ledger)
        if ($Workload.kind -eq "asr") { $args += @("--label", [string]$Workload.id) }
    }
    $native = Invoke-BenchmarkNativeJson -Exe $exe -Arguments $args -EchoOutput
    $result = $native.payload
    if (-not $native.succeeded) {
        $message = if ($result.error) { [string]$result.error } else { "executor exited with code $($native.exit_code)" }
        Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "executor-failed" $message $result $provenanceIds
        return $false
    }

    if ($Mode -eq "accuracy-quick") {
        $agreement = Measure-BenchmarkClassifierAgreement $result
        $intended = Test-BenchmarkIntendedProvider $result $RequestedDevice
        $eligible = -not ($ExecutionProfile.PSObject.Properties.Name -contains "accuracyEligible") -or
            [bool]$ExecutionProfile.accuracyEligible
        $valid = $agreement.outputs_finite -and $agreement.decision_flips -eq 0 -and
            $intended -and $eligible
        $record = New-BenchmarkAccuracyRecord -Workload $Workload `
            -ExecutionProfile $ExecutionProfile -RuntimeTarget $Runtime `
            -RequestedDevice $RequestedDevice `
            -RequestedProvider $(if ($Provider) { $Provider } else { "auto" }) `
            -ModelHash ([string]$result.model_sha256) -FixtureHash (Get-BenchmarkPathHash $path) `
            -GraphArtifacts $graphArtifacts `
            -EnvironmentSnapshotId $environmentSnapshot.id `
            -CompilationProvenanceIds $provenanceIds -Result $result `
            -ReferenceAgreement $agreement -Valid $valid -IntendedProviderResolved $intended
        Write-BenchmarkAccuracyRecord -Path $AccuracyLedger -Record $record
        Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice `
            $(if ($valid) { "accuracy-valid" } else { "accuracy-invalid" }) "" $record $provenanceIds
        return $valid
    }

    Write-SuiteAttempt $Workload $ExecutionProfile $RequestedDevice "ok" "" $result $provenanceIds
    return $true
}
