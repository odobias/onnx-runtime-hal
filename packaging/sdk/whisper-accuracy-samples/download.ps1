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

$jobs = foreach ($model in $models.PSObject.Properties) {
    $url = $model.Value
    $fileName = "$($model.Name)$($url.Substring($url.LastIndexOf('.')))"
    $target = Join-Path $outputDir $fileName

    Write-Host "Downloading $fileName from $url..."
    Start-Job -Name $fileName -ScriptBlock {
        param($url, $path)
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
    } -ArgumentList $url, $target
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
# serve. The URLs are immutable by convention, not by Artifactory policy.
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
