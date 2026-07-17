# Measurement-driven executor selection (-Executor fastest|most-accurate).

function Get-BenchmarkExecutorSelectionMetric {
    param(
        [Parameter(Mandatory)][ValidateSet("fastest", "most-accurate")][string]$Objective,
        [Parameter(Mandatory)][object[]]$Records
    )

    $valid = @($Records | Where-Object { $_.trust_gate.valid })
    if (-not $valid.Count) {
        return [pscustomobject]@{
            eligible = $false
            score = [double]::PositiveInfinity
            detail = "no accuracy-valid records"
        }
    }

    if ($Objective -eq "fastest") {
        $latencies = @()
        foreach ($rec in $valid) {
            $m = $rec.metrics
            $ms = if ($null -ne $m.median_ms) { [double]$m.median_ms }
            elseif ($null -ne $m.mean_ms) { [double]$m.mean_ms }
            else { $null }
            if ($null -eq $ms -or [double]::IsNaN($ms) -or [double]::IsInfinity($ms)) {
                return [pscustomobject]@{
                    eligible = $false
                    score = [double]::PositiveInfinity
                    detail = "missing latency for $($rec.workload_id)"
                }
            }
            $latencies += $ms
        }
        $score = ($latencies | Measure-Object -Sum).Sum
        return [pscustomobject]@{
            eligible = $true
            score = $score
            detail = ("sum_median_ms={0:N2}" -f $score)
            latency_ms = $score
        }
    }

    # most-accurate
    $asr = @($valid | Where-Object { $_.workload_kind -eq "asr" })
    $cls = @($valid | Where-Object { $_.workload_kind -ne "asr" })
    if ($asr.Count) {
        $wers = @()
        foreach ($rec in $asr) {
            $wer = $rec.metrics.wer
            if (-not (Test-BenchmarkFiniteNumber $wer)) {
                return [pscustomobject]@{
                    eligible = $false
                    score = [double]::PositiveInfinity
                    detail = "missing WER for $($rec.workload_id)"
                }
            }
            $wers += [double]$wer
        }
        $werScore = ($wers | Measure-Object -Average).Average
        $baselineWer = $null
        $werDelta = $null
        $first = $asr[0]
        if ($null -ne $first.metrics.baseline_wer) {
            $baselineWer = [double]$first.metrics.baseline_wer
            $werDelta = [double]$first.metrics.wer_delta
        }
        return [pscustomobject]@{
            eligible = $true
            score = $werScore
            detail = ("wer={0:P2}" -f $werScore)
            wer = $werScore
            baseline_wer = $baselineWer
            wer_delta = $werDelta
        }
    }

    $flips = 0
    foreach ($rec in $cls) {
        $flips += [int]$rec.reference_agreement.decision_flips
    }
    return [pscustomobject]@{
        eligible = $true
        score = [double]$flips
        detail = "decision_flips=$flips"
        decision_flips = $flips
    }
}

function Invoke-BenchmarkExecutorSelection {
    param(
        [Parameter(Mandatory)][ValidateSet("fastest", "most-accurate")][string]$Objective,
        [Parameter(Mandatory)][object]$PackageManifest,
        [Parameter(Mandatory)][string]$SuiteScript,
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string[]]$Device,
        [Parameter(Mandatory)][string[]]$Only,
        [Parameter(Mandatory)][string[]]$Profile,
        [Parameter(Mandatory)][string]$HostVendor,
        [string]$HostArchitecture = "",
        [ValidateSet("all", "bundled", "winml")][string]$Runtime = "all",
        [string]$Provider = "",
        [string]$Configuration = "Release",
        [string]$Precision = "default",
        [string]$Manifest = ""
    )

    $runtimes = if ($Runtime -eq "all") { @("bundled", "winml") } else { @($Runtime) }
    $pool = Get-BenchmarkExecutorCandidates -PackageManifest $PackageManifest `
        -Runtime $runtimes -Provider $Provider -HostVendor $HostVendor `
        -HostArchitecture $HostArchitecture
    if (-not $pool.candidates.Count) {
        throw "No packaged runners available for executor selection. $($pool.rejected -join '; ')"
    }

    $selectionId = [Guid]::NewGuid().ToString("N").Substring(0, 20)
    $outDir = Join-Path $Root "results\executor-selection"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $reportPath = Join-Path $outDir "$selectionId.json"
    $workDir = Join-Path $Root "build\cache\executor-selection\$selectionId"
    New-Item -ItemType Directory -Force -Path $workDir | Out-Null

    $requested = if ($Only -contains "all") { @("whisper", "tsc", "fakeaudio") } else { @($Only) }
    $trials = [System.Collections.Generic.List[object]]::new()
    $shellExe = (Get-Process -Id $PID).Path

    Write-Host ("executor selection: objective={0}, candidates={1}, devices={2}" -f `
        $Objective, $pool.candidates.Count, ($Device -join ",")) -ForegroundColor Cyan

    foreach ($candidate in @($pool.candidates | Sort-Object id, runtime_target)) {
        foreach ($dev in $Device) {
            $tag = "$($candidate.id)/$($candidate.runtime_target)/$dev"
            $accuracyLedger = Join-Path $workDir "$($candidate.id)-$($candidate.runtime_target)-$dev.accuracy.jsonl"
            $attemptLedger = Join-Path $workDir "$($candidate.id)-$($candidate.runtime_target)-$dev.attempts.jsonl"
            Remove-Item -LiteralPath $accuracyLedger, $attemptLedger -Force -ErrorAction SilentlyContinue

            Write-Host "`n--- trial $tag ---" -ForegroundColor Magenta
            $childArgs = [System.Collections.Generic.List[string]]::new()
            $childArgs.AddRange([string[]]@(
                "-NoLogo", "-NoProfile", "-File", $SuiteScript,
                "-Runtime", $candidate.runtime_target,
                "-RunnerId", $candidate.id,
                "-RunnerManifest", $PackageManifest.path,
                "-Device", $dev,
                "-Mode", "accuracy-quick",
                "-Configuration", $Configuration,
                "-Precision", $Precision,
                "-AccuracyLedger", $accuracyLedger,
                "-AttemptLedger", $attemptLedger,
                "-NoResults"
            ))
            foreach ($sel in $requested) { $childArgs.Add("-Only"); $childArgs.Add($sel) }
            foreach ($prof in @($Profile)) { $childArgs.Add("-Profile"); $childArgs.Add($prof) }
            if ($Provider) { $childArgs.Add("-Provider"); $childArgs.Add($Provider) }
            if ($Manifest) { $childArgs.Add("-Manifest"); $childArgs.Add($Manifest) }

            & $shellExe @($childArgs.ToArray())
            $exitCode = $LASTEXITCODE

            $records = @()
            if (Test-Path -LiteralPath $accuracyLedger) {
                $records = @(Get-Content -LiteralPath $accuracyLedger -Encoding UTF8 |
                    Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
            }
            $metric = Get-BenchmarkExecutorSelectionMetric -Objective $Objective -Records $records
            $expectedCount = $requested.Count
            # Profiles may expand; require at least one valid record per requested selector.
            $covered = @($records | ForEach-Object {
                if ($_.workload_kind -eq "asr") { "whisper" } else { [string]$_.workload_id }
            } | Select-Object -Unique)
            $missingSelectors = @($requested | Where-Object { $covered -notcontains $_ })
            $eligible = [bool]$metric.eligible -and -not $missingSelectors.Count -and
                (@($records | Where-Object { -not $_.trust_gate.valid }).Count -eq 0)

            $trials.Add([ordered]@{
                runner_id = $candidate.id
                runtime_target = $candidate.runtime_target
                device = $dev
                platform_tag = $candidate.platform_tag
                exit_code = $exitCode
                eligible = $eligible
                score = $metric.score
                detail = $metric.detail
                missing_selectors = $missingSelectors
                accuracy_ledger = $accuracyLedger.Replace($Root + '\', '').Replace('\', '/')
                records = @($records | ForEach-Object {
                    [ordered]@{
                        workload_id = $_.workload_id
                        execution_profile = $_.execution_profile
                        valid = [bool]$_.trust_gate.valid
                        resolved_provider = $_.resolved_provider
                        wer = $(if ($null -ne $_.metrics.wer) { $_.metrics.wer } else { $null })
                        baseline_wer = $(if ($null -ne $_.metrics.baseline_wer) { $_.metrics.baseline_wer } else { $null })
                        wer_delta = $(if ($null -ne $_.metrics.wer_delta) { $_.metrics.wer_delta } else { $null })
                        median_ms = $(if ($null -ne $_.metrics.median_ms) { $_.metrics.median_ms } elseif ($null -ne $_.metrics.mean_ms) { $_.metrics.mean_ms } else { $null })
                        decision_flips = $(if ($null -ne $_.reference_agreement.decision_flips) { [int]$_.reference_agreement.decision_flips } else { $null })
                    }
                })
            })
        }
    }

    $eligibleTrials = @($trials | Where-Object { $_.eligible } | Sort-Object score, runner_id, runtime_target, device)
    $winner = if ($eligibleTrials.Count) { $eligibleTrials[0] } else { $null }

    $report = [ordered]@{
        schema_version = 1
        objective = $Objective
        selection_id = $selectionId
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        requested = [ordered]@{
            devices = @($Device)
            only = @($requested)
            profiles = @($Profile)
            runtimes = @($runtimes)
            provider = $(if ($Provider) { $Provider } else { "auto" })
        }
        rejected_candidates = @($pool.rejected)
        trials = @($trials)
        winner = $(if ($winner) {
            [ordered]@{
                runner_id = $winner.runner_id
                runtime_target = $winner.runtime_target
                device = $winner.device
                platform_tag = $winner.platform_tag
                score = $winner.score
                detail = $winner.detail
            }
        } else { $null })
    }
    ($report | ConvertTo-Json -Depth 30) | Set-Content -LiteralPath $reportPath -Encoding UTF8

    Write-Host "`n==================== executor selection ====================" -ForegroundColor Magenta
    if ($winner) {
        Write-Host ("winner : {0}  runtime={1}  device={2}  ({3})" -f `
            $winner.runner_id, $winner.runtime_target, $winner.device, $winner.detail) -ForegroundColor Green
        if ($Objective -eq "most-accurate" -and $null -ne $winner.records) {
            $asr = @($winner.records | Where-Object { $null -ne $_.wer } | Select-Object -First 1)
            if ($asr -and $null -ne $asr.baseline_wer) {
                Write-Host ("baseline: WER={0:P2}  delta={1:+0.000%}" -f `
                    [double]$asr.baseline_wer, [double]$asr.wer_delta) -ForegroundColor DarkCyan
            }
        }
    } else {
        Write-Host "winner : (none - every candidate failed trust/validity gates)" -ForegroundColor Red
    }
    Write-Host "report : $reportPath" -ForegroundColor DarkGray
    return [pscustomobject]@{
        report_path = $reportPath
        winner = $winner
        trials = @($trials)
    }
}
