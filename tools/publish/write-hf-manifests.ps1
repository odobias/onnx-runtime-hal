<#
.SYNOPSIS
    Point the sdk model packages at the Hugging Face mirror, pinned to one
    immutable revision, and verify every file before writing the manifests.

.DESCRIPTION
    The Artifactory route needs a one-time 149 MB deploy into
    ai-models-generic-local, and that is blocked: reads there are anonymous but
    writes need a permission nobody has handed out yet. See
    tools/publish/upload-model-assets.ps1, which still implements that route if
    the grant ever arrives.

    Hugging Face needs no deploy at all, because the payload is ALREADY there and
    byte-identical to the local copy. What it needs instead is a credential, since
    gendigital/npu-hal-over-9000 is private and answers 401 to anonymous requests.
    That credential is cheap here: the fetch happens once per package version, on
    the sdk package build agent, and the resulting NuGet lands in Artifactory,
    which AvastClient's agents already read anonymously. So exactly one TeamCity
    job needs the secret rather than every consumer.

    URLs are pinned to a commit SHA, never to main. main is a mutable ref, and
    AvastClient's accuracy gate asserts a mean WER against ONE export -- a package
    that silently follows a re-upload would move that number and the failure would
    look like a code regression.

    Every file is verified before its URL is written. HF stores these as LFS
    objects and exposes each object's oid, which is a plain SHA-256 of the content,
    so the check is exact and needs no download. Anything HF serves as a normal git
    blob is fetched and hashed instead.

.PARAMETER Revision
    HF commit SHA to pin. Defaults to whatever main currently points at.

.PARAMETER Token
    HF token with read access. Falls back to $env:HF_TOKEN, then to the token the
    HF CLI caches in ~/.cache/huggingface/token.

.EXAMPLE
    .\tools\publish\write-hf-manifests.ps1
    .\tools\publish\write-hf-manifests.ps1 -Revision ccf1943146f34e9d2a56e5054a92d394dd97a675
#>

[CmdletBinding()]
param(
    [string]$Repo = "gendigital/npu-hal-over-9000",
    [string]$Revision,
    [string]$Token = $env:HF_TOKEN,
    [string]$HalRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($Token))
{
    $cached = Join-Path $env:USERPROFILE ".cache\huggingface\token"
    if (Test-Path -LiteralPath $cached)
    {
        $Token = (Get-Content -LiteralPath $cached -Raw).Trim()
    }
}
if ([string]::IsNullOrWhiteSpace($Token))
{
    throw ("No Hugging Face token. Set HF_TOKEN, pass -Token, or run 'hf auth login'. " +
        "$Repo is private, so anonymous requests get 401.")
}
$headers = @{ Authorization = "Bearer $Token" }

# The model package deliberately omits README.md, which HF carries but the
# package has no use for. The file list is the contract with SHA256SUMS.
$assets = @(
    @{
        Name = "whisper-tiny-multilingual-static"
        Package = "packaging\sdk\whisper-tiny-multilingual-static"
        LocalDir = "artifacts\workloads\whisper\models\static-onnx-tiny-multi-7s"
        HfPrefix = "whisper/static-onnx-tiny-multi-7s"
        Files = @(
            "encoder_model.onnx", "decoder_model.onnx", "vocab.json",
            "generation_config.json", "config.json", "preprocessor_config.json",
            "merges.txt", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "normalizer.json",
            "npu_hal_package.json"
        )
    },
    @{
        Name = "whisper-accuracy-samples"
        Package = "packaging\sdk\whisper-accuracy-samples"
        LocalDir = "artifacts\workloads\speech"
        HfPrefix = "speech"
        Files = @("ls_000.wav", "ls_001.wav", "ls_006.wav", "ls_008.wav", "ls_010.wav")
    }
)

if ([string]::IsNullOrWhiteSpace($Revision))
{
    $meta = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$Repo" -Headers $headers -TimeoutSec 60
    $Revision = $meta.sha
    Write-Host "pinning to current main: $Revision" -ForegroundColor DarkGray
}
if ($Revision -notmatch '^[0-9a-f]{40}$')
{
    throw "revision must be a full 40-character commit SHA, got '$Revision'"
}

# One tree listing per prefix, rather than one API call per file.
function Get-HfTree([string]$prefix)
{
    $url = "https://huggingface.co/api/models/$Repo/tree/$Revision/$prefix"
    $entries = Invoke-RestMethod -Uri $url -Headers $headers -TimeoutSec 90
    $map = @{}
    foreach ($e in $entries)
    {
        if ($e.type -ne "file") { continue }
        $map[(Split-Path $e.path -Leaf)] = $e
    }
    return $map
}

function Get-HfSha256([string]$prefix, $entry, [string]$name)
{
    if ($entry.lfs -and $entry.lfs.oid) { return $entry.lfs.oid.ToLowerInvariant() }

    # Not an LFS object, so HF exposes no content hash: fetch it and hash it. Only
    # small files land here, LFS covers anything sizeable.
    $tmp = New-TemporaryFile
    try
    {
        $url = "https://huggingface.co/$Repo/resolve/$Revision/$prefix/$name"
        Invoke-WebRequest -Uri $url -Headers $headers -OutFile $tmp -UseBasicParsing -TimeoutSec 300
        return (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    finally
    {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

$problems = @()
$plans = @()

foreach ($asset in $assets)
{
    Write-Host ""
    Write-Host "=== $($asset.Name) ===" -ForegroundColor Cyan

    $localDir = Join-Path $HalRoot $asset.LocalDir
    if (-not (Test-Path -LiteralPath $localDir))
    {
        $problems += "$($asset.Name): local payload not found at $localDir"
        continue
    }

    $tree = Get-HfTree $asset.HfPrefix
    $models = [ordered]@{}
    $sums = @()
    $downloaded = 0

    foreach ($name in $asset.Files)
    {
        $localFile = Join-Path $localDir $name
        if (-not (Test-Path -LiteralPath $localFile))
        {
            $problems += "$($asset.Name)/$name : missing locally"
            continue
        }
        $localSha = (Get-FileHash -LiteralPath $localFile -Algorithm SHA256).Hash.ToLowerInvariant()

        $entry = $tree[$name]
        if (-not $entry)
        {
            $problems += "$($asset.Name)/$name : not present on HF at $($Revision.Substring(0,12))"
            continue
        }

        $verifiedBy = if ($entry.lfs -and $entry.lfs.oid) { "lfs oid" } else { "download"; }
        $remoteSha = Get-HfSha256 $asset.HfPrefix $entry $name
        if (-not ($entry.lfs -and $entry.lfs.oid)) { $downloaded++ }

        if ($remoteSha -ne $localSha)
        {
            $problems += "$($asset.Name)/$name : HF content differs from local (HF $($remoteSha.Substring(0,12)), local $($localSha.Substring(0,12)))"
            Write-Host ("  {0,-28} MISMATCH" -f $name) -ForegroundColor Red
            continue
        }

        Write-Host ("  {0,-28} verified by {1}" -f $name, $verifiedBy) -ForegroundColor DarkGray

        # Key/URL contract of the shared sdk download.ps1: the key is the base
        # filename inside the package, the extension comes off the URL.
        $key = [IO.Path]::GetFileNameWithoutExtension($name)
        $models[$key] = "https://huggingface.co/$Repo/resolve/$Revision/$($asset.HfPrefix)/$name"
        $sums += "$localSha  models/$name"
    }

    $plans += [pscustomobject]@{
        Name = $asset.Name
        Package = Join-Path $HalRoot $asset.Package
        Models = $models
        Sums = $sums
        Verified = $asset.Files.Count
        Downloaded = $downloaded
    }
}

if ($problems.Count -gt 0)
{
    throw ("refusing to write manifests:`n  " + ($problems -join "`n  "))
}

Write-Host ""
foreach ($plan in $plans)
{
    $modelsJson = Join-Path $plan.Package "models.json"
    $sumsFile = Join-Path $plan.Package "SHA256SUMS"

    if ($WhatIfOnly)
    {
        Write-Host "$($plan.Name): would write $($plan.Models.Count) URL(s) to $modelsJson" -ForegroundColor Yellow
        continue
    }

    # UTF-8 without BOM: build.cmd reads these under Windows PowerShell 5.1, whose
    # ConvertFrom-Json chokes on a BOM.
    $json = ($plan.Models | ConvertTo-Json -Depth 3)
    [IO.File]::WriteAllText($modelsJson, $json + "`r`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($sumsFile, (($plan.Sums -join "`r`n") + "`r`n"), [Text.UTF8Encoding]::new($false))

    Write-Host "$($plan.Name): $($plan.Models.Count) file(s) pinned at $($Revision.Substring(0,12))" -ForegroundColor Green
    Write-Host "  -> $modelsJson"
    Write-Host "  -> $sumsFile"
}

Write-Host ""
Write-Host "revision $Revision" -ForegroundColor DarkGray
Write-Host ("verified without downloading: {0} of {1} file(s)" -f
    (($plans | Measure-Object -Property Verified -Sum).Sum - ($plans | Measure-Object -Property Downloaded -Sum).Sum),
    ($plans | Measure-Object -Property Verified -Sum).Sum) -ForegroundColor DarkGray
