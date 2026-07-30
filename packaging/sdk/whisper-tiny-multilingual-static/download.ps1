# Downloads every file listed in models.json into .\models and verifies it
# against SHA256SUMS.
#
# This is a stricter variant of the download.ps1 the other sdk model packages
# share. Theirs starts a background job per file and then discards the job
# results without inspecting them, so a 404, a proxy error or a truncated
# transfer still leaves the build "successful" and publishes a package with a
# missing or half-written model in it. The payload here is 148 MB and it backs
# an accuracy gate, so a quietly incomplete package is worse than a red build.
#
# Key/URL contract, unchanged: the key is the base filename inside the package
# and the extension comes from the URL.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path -Path $scriptDir -ChildPath "models"

if (-not (Test-Path -Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$models = Get-Content -Path (Join-Path $scriptDir "models.json") -Raw | ConvertFrom-Json

# The payload lives in a PRIVATE Hugging Face repo, so unlike the other sdk model
# packages this one needs a credential. Anonymous requests get 401, and because
# every URL here is pinned to a commit SHA rather than to main, a package built
# today fetches exactly the export the WER baselines were measured against.
#
# The token is only needed HERE, on the package build agent. The resulting NuGet
# goes to Artifactory, which the consuming build agents already read anonymously.
$needsAuth = @($models.PSObject.Properties | Where-Object { $_.Value -like "https://huggingface.co/*" }).Count -gt 0
$authHeader = @{}
if ($needsAuth) {
    $token = $env:HF_TOKEN
    if (-not $token) {
        $cached = Join-Path $env:USERPROFILE ".cache\huggingface\token"
        if (Test-Path -Path $cached) { $token = (Get-Content -Path $cached -Raw).Trim() }
    }
    if (-not $token) {
        throw ("models.json points at the private repo gendigital/npu-hal-over-9000, " +
            "which returns 401 without a credential. Set the HF_TOKEN environment " +
            "variable to a token with read access to it. In TeamCity this belongs in " +
            "a secure parameter exposed as env.HF_TOKEN, not in this repository.")
    }
    $authHeader = @{ Authorization = "Bearer $token" }
    Write-Host "Using HF_TOKEN for huggingface.co downloads."
}

$jobs = foreach ($model in $models.PSObject.Properties) {
    $url = $model.Value
    $fileName = "$($model.Name)$($url.Substring($url.LastIndexOf('.')))"
    $target = Join-Path $outputDir $fileName

    Write-Host "Downloading $fileName from $url..."
    Start-Job -Name $fileName -ScriptBlock {
        param($url, $path, $headers)
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -Headers $headers -OutFile $path -UseBasicParsing
    } -ArgumentList $url, $target, $authHeader
}

$failed = @()
foreach ($job in $jobs) {
    Wait-Job -Job $job | Out-Null
    Receive-Job -Job $job -ErrorAction SilentlyContinue -ErrorVariable jobError | Out-Null
    if ($job.State -ne "Completed" -or $jobError) {
        $failed += "$($job.Name): $(($jobError | Select-Object -First 1))"
    }
    Remove-Job -Job $job
}

if ($failed) {
    throw "download failed:`n  " + ($failed -join "`n  ")
}

# Fail loudly on drift between the committed checksums and what the URLs now
# serve. The URLs pin a commit SHA, so this should never fire -- which is exactly
# why it is worth checking. A silent substitution here would surface much later as
# an unexplained WER regression in a completely different repository.
$sumsFile = Join-Path $scriptDir "SHA256SUMS"
if (Test-Path -Path $sumsFile) {
    $mismatched = @()
    foreach ($line in Get-Content -Path $sumsFile) {
        if ($line -notmatch '^([0-9a-fA-F]{64})\s+(?:models/)?(.+)$') { continue }
        $expected = $Matches[1].ToLower()
        $path = Join-Path $outputDir $Matches[2]
        if (-not (Test-Path -Path $path)) {
            $mismatched += "$($Matches[2]): not downloaded"
            continue
        }
        $actual = (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $expected) {
            $mismatched += "$($Matches[2]): expected $expected, got $actual"
        }
    }
    if ($mismatched) {
        throw "checksum mismatch:`n  " + ($mismatched -join "`n  ")
    }
    Write-Host "Checksums verified against SHA256SUMS."
} else {
    Write-Warning "no SHA256SUMS next to models.json; downloads are unverified."
}

Write-Host "All files downloaded successfully to $outputDir."
