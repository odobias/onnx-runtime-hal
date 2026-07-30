<#
.SYNOPSIS
    Upload the Whisper static package and accuracy corpus to Artifactory, and
    write the models.json that the sdk package repos download them with.

.DESCRIPTION
    The sdk model packages under git.int.avast.com/sdk fetch their payload with
    a shared download.ps1 that does a plain unauthenticated Invoke-WebRequest
    against ai-models-generic-local. Reads from that repository are anonymous,
    so the packages need no credentials -- but getting the bytes IN does, which
    is why this step is separate from the package build.

    Most of ai-models-generic-local is written by Vertex AI pipelines, under
    vertex-ai/<project>/<region>/. Ours are not pipeline outputs, so they go at
    the top level next to the other hand-uploaded payload,
    ai-models-generic-local/sherpa-onnx/whisper/small/1/.

    Paths include the short commit of the HAL checkout that produced the bytes.
    That makes a re-export land beside the old copy instead of silently
    replacing it, which matters because the WER baselines committed in
    AvastClient are only meaningful against one specific export.

    Uploads are skipped when a file of the same SHA-256 is already at the
    target, so re-running after a partial failure is cheap. 155 MB over a VPN
    is not something to repeat for fun.

.PARAMETER Token
    Artifactory identity token or API key with deploy rights on
    ai-models-generic-local. Falls back to $env:ARTIFACTORY_TOKEN.

.PARAMETER DryRun
    Report what would be uploaded and write models.json, but transfer nothing.
    Useful for reviewing the URLs before committing to them.

.EXAMPLE
    $env:ARTIFACTORY_TOKEN = "<token>"
    .\tools\publish\upload-model-assets.ps1 -DryRun
    .\tools\publish\upload-model-assets.ps1
#>

[CmdletBinding()]
param(
    [string]$Token = $env:ARTIFACTORY_TOKEN,
    [string]$HalRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$BaseUrl = "https://artifactory.ida.avast.com/artifactory",
    [string]$Repository = "ai-models-generic-local",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$ProgressPreference = "SilentlyContinue"

if (-not $DryRun -and [string]::IsNullOrWhiteSpace($Token))
{
    throw "No Artifactory token. Pass -Token or set ARTIFACTORY_TOKEN, or use -DryRun."
}

Push-Location $HalRoot
try
{
    $commit = (& git rev-parse --short HEAD).Trim()
    $dirty = [bool](& git status --porcelain -- artifacts/workloads)
}
finally
{
    Pop-Location
}

if ($dirty)
{
    Write-Warning ("artifacts/workloads has uncommitted changes; the bytes you " +
        "upload will not match commit $commit that names them.")
}

# The 4 files StaticOrtSession.cpp actually opens, plus the rest of the HF
# export. The extras are 4.4 MB against 148 MB and are what a future
# re-export or tokenizer investigation needs, so leaving them out saves
# nothing and costs traceability.
$modelFiles = @(
    "encoder_model.onnx", "decoder_model.onnx", "vocab.json",
    "generation_config.json", "config.json", "preprocessor_config.json",
    "merges.txt", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "normalizer.json",
    "npu_hal_package.json"
)

# Keep in step with kAccuracyCorpus in AvastClient
# framework/whisper/src/whisper_unit_test/AccuracyCorpus.h.
$sampleFiles = @("ls_000.wav", "ls_001.wav", "ls_006.wav", "ls_008.wav", "ls_010.wav")

$assets = @(
    @{
        Name = "static-onnx-tiny-multi-7s"
        From = "artifacts\workloads\whisper\models\static-onnx-tiny-multi-7s"
        Files = $modelFiles
        Package = "packaging\sdk\whisper-tiny-multilingual-static"
    }
    @{
        Name = "accuracy-samples"
        From = "artifacts\workloads\speech"
        Files = $sampleFiles
        Package = "packaging\sdk\whisper-accuracy-samples"
    }
)

function Get-RemoteSha256([string]$url)
{
    try
    {
        $info = Invoke-RestMethod -Uri $url -TimeoutSec 60
        return $info.checksums.sha256
    }
    catch
    {
        return $null
    }
}

$uploaded = 0
$reused = 0
$bytes = 0L

foreach ($asset in $assets)
{
    $sourceDir = Join-Path $HalRoot $asset.From
    if (-not (Test-Path -LiteralPath $sourceDir))
    {
        throw "asset source missing: $sourceDir"
    }

    Write-Host ""
    Write-Host ("=== {0} @ {1}" -f $asset.Name, $commit) -ForegroundColor Cyan

    $map = [ordered]@{}
    $sums = @()

    foreach ($name in $asset.Files)
    {
        $source = Join-Path $sourceDir $name
        if (-not (Test-Path -LiteralPath $source))
        {
            throw "$($asset.Name): $name not in the HAL tree ($source)"
        }

        $length = (Get-Item -LiteralPath $source).Length
        $sha = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()
        $target = "$BaseUrl/$Repository/whisper-npu-hal/$($asset.Name)/$commit/$name"

        # download.ps1 derives the on-disk name as <key><extension-of-url>, so
        # the key must NOT carry the extension. sdk/sherpa-onnx-whisper-small
        # shipped 0.0.1 with keys like "small-encoder.onnx", writing
        # small-encoder.onnx.onnx into the package; branch 0.0.2 exists only to
        # strip them again. Deriving the key here keeps that unrepeatable.
        $key = [IO.Path]::GetFileNameWithoutExtension($name)
        $map[$key] = $target
        $sums += ("{0}  models/{1}" -f $sha, $name)

        $existing = Get-RemoteSha256 "$BaseUrl/api/storage/$Repository/whisper-npu-hal/$($asset.Name)/$commit/$name"
        if ($existing -eq $sha)
        {
            Write-Host ("  = {0} ({1:N2} MB, already deployed)" -f $name, ($length / 1MB)) -ForegroundColor DarkGray
            $reused++
            continue
        }
        if ($existing)
        {
            throw ("$name exists at $target with a different checksum. " +
                "Refusing to overwrite; upload under a fresh commit instead.")
        }

        if ($DryRun)
        {
            Write-Host ("  would upload {0} ({1:N2} MB)" -f $name, ($length / 1MB)) -ForegroundColor Yellow
        }
        else
        {
            # Artifactory verifies the checksum it is told to expect, so a
            # truncated transfer fails the PUT instead of poisoning the feed.
            Invoke-RestMethod -Uri $target -Method Put -InFile $source `
                -Headers @{ "Authorization" = "Bearer $Token"; "X-Checksum-Sha256" = $sha } `
                -ContentType "application/octet-stream" -TimeoutSec 1800 | Out-Null
            Write-Host ("  + {0} ({1:N2} MB)" -f $name, ($length / 1MB)) -ForegroundColor Green
        }
        $uploaded++
        $bytes += $length
    }

    $packageDir = Join-Path $HalRoot $asset.Package
    if (Test-Path -LiteralPath $packageDir)
    {
        $json = ($map | ConvertTo-Json) -replace "`r`n", "`n"
        Set-Content -LiteralPath (Join-Path $packageDir "models.json") -Value $json -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $packageDir "SHA256SUMS") -Value $sums -Encoding UTF8
        Write-Host ("  -> {0}\models.json, SHA256SUMS" -f $asset.Package)
    }
    else
    {
        Write-Warning "no package dir at $packageDir; models.json not written"
    }
}

Write-Host ""
Write-Host ("{0} file(s) {1}, {2:N1} MB{3}" -f $uploaded,
    $(if ($DryRun) { "to upload" } else { "uploaded" }), ($bytes / 1MB),
    $(if ($reused) { " ($reused already deployed)" } else { "" })) -ForegroundColor Green

if ($uploaded -and -not $DryRun)
{
    Write-Host ""
    Write-Host "Commit the regenerated models.json, then push each packaging/sdk" -ForegroundColor Cyan
    Write-Host "subtree to its sdk repo on a branch named for the package version."
}
