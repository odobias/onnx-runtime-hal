<#
.SYNOPSIS
    Upload the Whisper static package and accuracy corpus to Artifactory, and
    write the models.json that the sdk package repos download them with.

    NOT the route currently in use. The packages fetch from Hugging Face instead;
    see tools/publish/write-hf-manifests.ps1 and packaging/sdk/README.md. This
    script is kept because the Artifactory route is preferable in one respect --
    reads are anonymous, so no build job needs a secret -- and it becomes usable
    the moment somebody grants deploy on ai-models-generic-local. Nothing here is
    stale: the preflight and the content-addressed paths were both exercised
    against the live server.

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

    Upload paths are content-addressed: a short digest over the payload's own
    SHA-256 list, so a re-export lands beside the old copy instead of silently
    replacing it. That matters because the WER baselines committed in
    AvastClient are only meaningful against one specific export.

    Uploads are skipped when a file of the same SHA-256 is already at the
    target, so re-running after a partial failure is cheap. 155 MB over a VPN
    is not something to repeat for fun.

.PARAMETER Token
    Artifactory identity token or API key with deploy rights on
    ai-models-generic-local. Falls back to $env:ARTIFACTORY_TOKEN, then to the
    first line of the file named by -TokenFile.

.PARAMETER TokenFile
    Read the token from this file instead of passing it on a command line, where
    it would end up in shell history and in any transcript of the session.
    Defaults to %USERPROFILE%\.artifactory-token if that exists.

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
    [string]$TokenFile = (Join-Path $env:USERPROFILE ".artifactory-token"),
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

if ([string]::IsNullOrWhiteSpace($Token) -and (Test-Path -LiteralPath $TokenFile))
{
    $Token = (Get-Content -LiteralPath $TokenFile -TotalCount 1).Trim()
}
if (-not $DryRun -and [string]::IsNullOrWhiteSpace($Token))
{
    throw ("No Artifactory token. Write one to $TokenFile, or pass -Token, or " +
        "set ARTIFACTORY_TOKEN, or use -DryRun.")
}

# Name the upload directory after the payload itself: SHA-256 over the sorted
# "<file>:<sha256>" list, truncated. Not after a git commit, which was the
# obvious first choice and is wrong twice over -- /artifacts/** is gitignored, so
# no commit describes these bytes at all, and HEAD would rewrite every URL
# whenever an unrelated file changed.
#
# Content addressing gives exactly the property the WER baselines need: the path
# changes when and only when the payload does, so a re-export cannot land on top
# of the export a committed baseline was measured against.
function Get-PayloadDigest([hashtable]$shaByName)
{
    $lines = $shaByName.Keys | Sort-Object | ForEach-Object { "${_}:$($shaByName[$_])" }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try
    {
        return (($sha.ComputeHash($bytes) | ForEach-Object { '{0:x2}' -f $_ }) -join '').Substring(0, 12)
    }
    finally
    {
        $sha.Dispose()
    }
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

# Prove the token can actually deploy before moving 149 MB. Artifactory has no
# read-only way to ask "may I write here", and a token that authenticates fine
# can still lack deploy rights on this repository, so the check is a tiny PUT
# followed by a DELETE. Failing this costs a second; failing on byte 100 million
# of a 113 MB file over a VPN does not.
function Test-DeployAccess([string]$token)
{
    $headers = @{ "Authorization" = "Bearer $token" }

    # Ask who we are FIRST, because a 403 on its own cannot tell "authenticated
    # but unprivileged" from "credential ignored". This instance does not answer
    # 401 for a bad token, and it lets anonymous read /api/repositories and even
    # /api/security/encryptedPassword -- so a 200 from those endpoints is no
    # evidence of having authenticated. Measured: a real token, deliberate
    # rubbish and no header at all produce identical results on every one of them.
    #
    # This endpoint reports the principal's name, which is the one thing that
    # actually distinguishes the two cases.
    $identityUrl = ($BaseUrl -replace '/artifactory/?$', '') + "/ui/api/v1/ui/auth/current"
    $who = $null
    try
    {
        $who = (Invoke-RestMethod -Uri $identityUrl -Headers $headers -TimeoutSec 30).name
    }
    catch
    {
        Write-Warning ("could not determine the authenticated identity ({0}); " -f $_.Exception.Message.Trim() +
            "continuing to the deploy probe, which is the authoritative check anyway")
    }

    if ($who -and $who -ne "anonymous")
    {
        Write-Host "preflight: authenticated as $who" -ForegroundColor DarkGray
    }
    elseif ($who -eq "anonymous")
    {
        throw ("preflight failed: Artifactory resolves this token to 'anonymous', " +
            "so the credential is not being accepted at all -- it is expired, revoked, " +
            "or was issued by a different Artifactory host than $BaseUrl. Note that a " +
            "well-formed token is not enough: a reference token still decodes to " +
            "'reftkn:01...' long after it stops working, and this instance answers 403 " +
            "rather than 401, so nothing else here will tell you.")
    }

    $probe = "$BaseUrl/$Repository/whisper-npu-hal/.preflight-$([guid]::NewGuid().ToString('N'))"
    try
    {
        Invoke-RestMethod -Uri $probe -Method Put -Body "preflight" `
            -Headers $headers -ContentType "text/plain" -TimeoutSec 60 | Out-Null
    }
    catch
    {
        $code = $_.Exception.Response.StatusCode.value__
        # Artifactory answers an unrecognised token by falling back to anonymous
        # rather than by rejecting it, so a typo arrives here as 403 and not 401.
        # Do not claim the token authenticated.
        $hint = switch ($code)
        {
            401 { "the token was rejected outright; generate a fresh one" }
            403 { "authenticated, but without deploy rights on $Repository. The " +
                  "identity check above passed, so this is a missing permission " +
                  "and not a bad credential: someone with admin on $Repository " +
                  "has to grant deploy" }
            default { $_.Exception.Message }
        }
        throw "preflight failed (HTTP $code): $hint"
    }

    try
    {
        Invoke-RestMethod -Uri $probe -Method Delete -Headers $headers -TimeoutSec 60 | Out-Null
    }
    catch
    {
        # Deploy works, which is what the payload needs. Leaving a marker behind
        # is untidy but not a reason to stop.
        Write-Warning "preflight marker could not be removed: $probe"
    }
}

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

if (-not $DryRun)
{
    Test-DeployAccess $Token
    Write-Host "preflight: deploy to $Repository confirmed" -ForegroundColor DarkGray
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

    # First pass: hash everything, because the directory name is derived from
    # the whole payload and nothing can be uploaded before it is known.
    $shaByName = @{}
    $lengthByName = @{}
    foreach ($name in $asset.Files)
    {
        $source = Join-Path $sourceDir $name
        if (-not (Test-Path -LiteralPath $source))
        {
            throw "$($asset.Name): $name not in the HAL tree ($source)"
        }
        $shaByName[$name] = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()
        $lengthByName[$name] = (Get-Item -LiteralPath $source).Length
    }

    $digest = Get-PayloadDigest $shaByName

    Write-Host ""
    Write-Host ("=== {0} @ {1} ({2} file(s))" -f $asset.Name, $digest, $asset.Files.Count) `
        -ForegroundColor Cyan

    $map = [ordered]@{}
    $sums = @()

    foreach ($name in $asset.Files)
    {
        $source = Join-Path $sourceDir $name
        $length = $lengthByName[$name]
        $sha = $shaByName[$name]
        $target = "$BaseUrl/$Repository/whisper-npu-hal/$($asset.Name)/$digest/$name"

        # download.ps1 derives the on-disk name as <key><extension-of-url>, so
        # the key must NOT carry the extension. sdk/sherpa-onnx-whisper-small
        # shipped 0.0.1 with keys like "small-encoder.onnx", writing
        # small-encoder.onnx.onnx into the package; branch 0.0.2 exists only to
        # strip them again. Deriving the key here keeps that unrepeatable.
        $key = [IO.Path]::GetFileNameWithoutExtension($name)
        $map[$key] = $target
        $sums += ("{0}  models/{1}" -f $sha, $name)

        $existing = Get-RemoteSha256 "$BaseUrl/api/storage/$Repository/whisper-npu-hal/$($asset.Name)/$digest/$name"
        if ($existing -eq $sha)
        {
            Write-Host ("  = {0} ({1:N2} MB, already deployed)" -f $name, ($length / 1MB)) -ForegroundColor DarkGray
            $reused++
            continue
        }
        if ($existing)
        {
            # Content addressing should make this unreachable: different bytes
            # produce a different directory. If it fires, the digest scheme is
            # broken, so stop rather than overwrite something a baseline cites.
            throw ("$name exists at $target with a DIFFERENT checksum, which " +
                "should be impossible under content addressing. Refusing to " +
                "overwrite. Expected $sha, found $existing.")
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
