# Downloads the pinned multilingual Whisper corpus from Hugging Face into
# .\models, verifies every WAV against SHA256SUMS, then copies the committed
# accuracy contract alongside it.
#
# Key/URL contract: the key is the output base filename and the extension comes
# from the URL. The HF URLs pin an immutable revision; never point this at main.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $scriptDir "models"
$modelsJson = Join-Path $scriptDir "models.json"

if (Test-Path -LiteralPath $outputDir) {
    Remove-Item -LiteralPath $outputDir -Recurse -Force
}
New-Item -ItemType Directory -Path $outputDir | Out-Null

$models = Get-Content -LiteralPath $modelsJson -Raw | ConvertFrom-Json
$props = @($models.PSObject.Properties)
if ($props.Count -eq 0) {
    throw "models.json is empty -- nothing to pack"
}

$authHeader = @{}
if (@($props | Where-Object { $_.Value -like "https://huggingface.co/*" }).Count -gt 0) {
    $token = $env:HF_TOKEN
    if (-not $token) {
        $cached = Join-Path $env:USERPROFILE ".cache\huggingface\token"
        if (Test-Path -LiteralPath $cached) {
            $token = (Get-Content -LiteralPath $cached -Raw).Trim()
        }
    }
    if (-not $token) {
        throw ("models.json points at the private repo gendigital/npu-hal-over-9000. " +
            "Set HF_TOKEN to a token with read access. In TeamCity, expose it as " +
            "the secure parameter env.HF_TOKEN.")
    }
    $authHeader = @{ Authorization = "Bearer $token" }
    Write-Host "Using HF_TOKEN for huggingface.co downloads."
}

foreach ($model in $props) {
    $url = [string]$model.Value
    $fileName = "$($model.Name)$($url.Substring($url.LastIndexOf('.')))"
    $target = Join-Path $outputDir $fileName
    Write-Host "Downloading $fileName from $url..."
    Invoke-WebRequest -Uri $url -Headers $authHeader -OutFile $target -UseBasicParsing
}

$sumsFile = Join-Path $scriptDir "SHA256SUMS"
if (Test-Path -LiteralPath $sumsFile) {
    $mismatched = @()
    foreach ($line in Get-Content -LiteralPath $sumsFile) {
        if ($line -notmatch '^([0-9a-fA-F]{64})\s+(?:models/)?(.+)$') { continue }
        $expected = $Matches[1].ToLower()
        $path = Join-Path $outputDir $Matches[2]
        if (-not (Test-Path -LiteralPath $path)) {
            $mismatched += "$($Matches[2]): missing"
            continue
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $expected) {
            $mismatched += "$($Matches[2]): expected $expected, got $actual"
        }
    }
    if ($mismatched) {
        throw "checksum mismatch:`n  " + ($mismatched -join "`n  ")
    }
    Write-Host "Checksums verified against SHA256SUMS."
}
else {
    Write-Warning "no SHA256SUMS next to models.json; downloads are unverified."
}

$contract = Join-Path $scriptDir "samples.json"
if (-not (Test-Path -LiteralPath $contract)) {
    throw "samples.json is missing -- the WAVs have no committed accuracy contract"
}
Copy-Item -LiteralPath $contract -Destination (Join-Path $outputDir "samples.json")

Write-Host "All files downloaded and verified at $outputDir."
