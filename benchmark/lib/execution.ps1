# Benchmark library module. Dot-source benchmark/lib/harness.ps1 instead of loading this directly.

function Invoke-BenchmarkNativeJson {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$JsonOutputPath = "",
        [switch]$EchoOutput
    )
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $raw = [System.Collections.Generic.List[object]]::new()
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $raw.Add($_)
            if ($EchoOutput) { Write-Host ([string]$_) }
        }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldEap
    }
    $payload = $null
    if ($JsonOutputPath -and (Test-Path -LiteralPath $JsonOutputPath)) {
        try {
            $payload = Get-Content -LiteralPath $JsonOutputPath -Raw -Encoding UTF8 |
                ConvertFrom-Json -ErrorAction Stop
        } catch {}
    }
    for ($i = $raw.Count - 1; $i -ge 0; --$i) {
        if ($payload) { break }
        try {
            $candidate = ([string]$raw[$i]) | ConvertFrom-Json -ErrorAction Stop
            if ($null -ne $candidate.ok) { $payload = $candidate; break }
        } catch {}
    }
    if (-not $payload) {
        $payload = [pscustomobject]@{ ok = $false; error = "no-json-output (exit $exitCode)" }
    }
    return [pscustomobject]@{
        exit_code = $exitCode
        payload = $payload
        output = $raw
        succeeded = ($exitCode -eq 0 -and $payload.ok)
    }
}

function Invoke-BenchmarkClip {
    param(
        [Parameter(Mandatory)] [string]$Exe,
        [Parameter(Mandatory)] [string]$ModelDir,
        [Parameter(Mandatory)] [string]$Audio,
        [Parameter(Mandatory)] [string]$Backend,
        [Parameter(Mandatory)] [string]$Device,
        [Parameter(Mandatory)] [int]$Runs,
        [Parameter(Mandatory)] [string]$CacheDir,
        [Parameter(Mandatory)] [string]$Ref,
        [int]$Threads = 0,
        [string]$Provider = "",
        [switch]$HotOnly,
        [switch]$EchoOutput
    )
    $a = @("run", "whisper", $ModelDir, $Audio, $Backend, $Device, "$Runs", "--cache", $CacheDir, "--ref", $Ref, "--json")
    if ($Threads -gt 0) { $a += @("--threads", "$Threads") }
    if ($Provider) { $a += @("--provider", $Provider) }
    if ($HotOnly) { $a += "--hot-only" }

    return (Invoke-BenchmarkNativeJson -Exe $Exe -Arguments $a -EchoOutput:$EchoOutput).payload
}

# --- aggregation -------------------------------------------------------------

# Cold/hot engine-load seconds come from the first successful clip's record
# (load happens once per engine, not per clip). Tolerates the legacy load_warm_s
# field name for hot start.
function Get-BenchmarkLoadTimes($FirstRow) {
    $cold = Get-BenchmarkOptional $FirstRow "cold_load_seconds" `
        (Get-BenchmarkOptional $FirstRow "load_cold_s" -1)
    $hot = Get-BenchmarkOptional $FirstRow "hot_load_seconds" `
        (Get-BenchmarkOptional $FirstRow "load_hot_s" `
            (Get-BenchmarkOptional $FirstRow "warm_load_seconds" -1))
    return [pscustomobject]@{ cold = $cold; hot = $hot }
}

# Aggregate a set of successful clip records into the means/sums both harnesses
# report. WER/CER are micro-averaged (sum edits / sum refs), not averaged per clip.
function Measure-BenchmarkClips($Rows) {
    $wEdits = ($Rows | Measure-Object word_edits -Sum).Sum
    $wRef = ($Rows | Measure-Object ref_words -Sum).Sum
    $cEdits = ($Rows | Measure-Object char_edits -Sum).Sum
    $cRef = ($Rows | Measure-Object ref_chars -Sum).Sum
    return [pscustomobject]@{
        count         = $Rows.Count
        mean_ms       = ($Rows | Measure-Object mean_ms -Average).Average
        mean_rtf      = ($Rows | Measure-Object rtf -Average).Average
        mean_tps      = ($Rows | Measure-Object throughput_tps -Average).Average
        mean_logprob  = ($Rows | Measure-Object avg_logprob -Average).Average
        mean_ttft_ms  = ($Rows | Measure-Object ttft_ms -Average).Average
        mean_tpot_ms  = ($Rows | Measure-Object tpot_ms -Average).Average
        audio_seconds = ($Rows | Measure-Object audio_len_s -Sum).Sum
        word_edits    = $wEdits
        ref_words     = $wRef
        char_edits    = $cEdits
        ref_chars     = $cRef
        wer           = if ($wRef) { $wEdits / $wRef } else { $null }
        cer           = if ($cRef) { $cEdits / $cRef } else { $null }
        first_row     = ($Rows | Select-Object -First 1)
        last_row      = ($Rows | Select-Object -Last 1)
    }
}
