<#
.SYNOPSIS
  Mirror one immutable HAL source package release into a NuGet feed.

.DESCRIPTION
  Downloads the package, SHA-256 sidecar, and provenance from the private
  GitHub release, verifies that all three agree, then pushes the original
  .nupkg bytes unchanged. Intended for a trusted downstream CI mirror job.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,

    [Parameter(Mandatory = $true)]
    [string]$NuGetSource,

    [string]$Repository = "odobias/onnx-runtime-hal",

    [string]$ApiKey = $env:NUGET_API_KEY,

    [string]$Output = "",

    [switch]$SkipPush
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:DOTNET_CLI_UI_LANGUAGE = "en"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $Output) {
    $Output = Join-Path $root "artifacts\mirror\onnx-runtime-hal-source\$Tag"
}
$Output = [IO.Path]::GetFullPath($Output)

if (Test-Path -LiteralPath $Output) {
    Remove-Item -LiteralPath $Output -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Output | Out-Null

& gh release download $Tag `
    --repo $Repository `
    --dir $Output `
    --pattern "GenDigital.OnnxRuntimeHal.Source.*.nupkg" `
    --pattern "GenDigital.OnnxRuntimeHal.Source.*.nupkg.sha256" `
    --pattern "GenDigital.OnnxRuntimeHal.Source.*.nupkg.provenance.json"
if ($LASTEXITCODE -ne 0) {
    throw "GitHub release download failed."
}

$packages = @(Get-ChildItem -LiteralPath $Output -Filter "GenDigital.OnnxRuntimeHal.Source.*.nupkg" -File)
if ($packages.Count -ne 1) {
    throw "Expected exactly one .nupkg, found $($packages.Count)."
}
$package = $packages[0]
$shaPath = "$($package.FullName).sha256"
$provenancePath = "$($package.FullName).provenance.json"
if (-not (Test-Path -LiteralPath $shaPath) -or
    -not (Test-Path -LiteralPath $provenancePath)) {
    throw "The release is missing its SHA-256 or provenance sidecar."
}

$expectedSha = ((Get-Content -LiteralPath $shaPath -Raw -Encoding utf8).Trim() -split '\s+')[0]
$actualSha = (Get-FileHash -LiteralPath $package.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
if ($expectedSha -ne $actualSha) {
    throw "Package SHA-256 mismatch: expected $expectedSha, got $actualSha."
}

$provenance = Get-Content -LiteralPath $provenancePath -Raw -Encoding utf8 | ConvertFrom-Json
if ($provenance.package_file -ne $package.Name -or $provenance.sha256 -ne $actualSha) {
    throw "Package provenance does not match the downloaded package."
}
if ($Tag -notmatch '^onnx-runtime-hal-source-v(?<version>.+)$' -or
    $provenance.package_version -ne $Matches.version) {
    throw "Release tag '$Tag' does not match provenance version '$($provenance.package_version)'."
}
if ($provenance.dirty_source) {
    throw "Refusing to mirror a package produced from a dirty source tree."
}

if ($SkipPush) {
    Write-Host "Verified $($package.Name); push skipped." -ForegroundColor Green
    return
}
if (-not $ApiKey) {
    throw "NuGet API key is required through -ApiKey or NUGET_API_KEY."
}

& dotnet nuget push $package.FullName `
    --source $NuGetSource `
    --api-key $ApiKey `
    --skip-duplicate
if ($LASTEXITCODE -ne 0) {
    throw "NuGet mirror push failed."
}

Write-Host "Mirrored $($package.Name) unchanged to $NuGetSource" -ForegroundColor Green
