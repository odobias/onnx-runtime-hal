<#
.SYNOPSIS
  Build the immutable GenDigital.OnnxRuntimeHal.Source NuGet package.

.DESCRIPTION
  Packs workload-neutral HAL headers, implementation sources, and native
  MSBuild integration. The package contains no model or runtime binaries.
  A SHA-256 sidecar and deterministic provenance document are emitted beside
  the .nupkg for GitHub publication and downstream feed mirroring.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$')]
    [string]$Version,

    [string]$Output = "",

    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:DOTNET_CLI_UI_LANGUAGE = "en"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$project = Join-Path $root "packaging\onnx-runtime-hal-source\OnnxRuntimeHal.Source.csproj"
if (-not $Output) {
    $Output = Join-Path $root "artifacts\dist\onnx-runtime-hal-source"
}
$Output = [IO.Path]::GetFullPath($Output)

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    $out = & git -C $root @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed."
    }
    return $out
}

$dirty = [bool](Invoke-Git status --porcelain)
if ($dirty -and -not $AllowDirty) {
    throw "The source tree is dirty. Commit the package inputs or use -AllowDirty for local rehearsal."
}

$commit = (Invoke-Git rev-parse HEAD).Trim()
$commitTime = (Invoke-Git show -s --format=%cI HEAD).Trim()

New-Item -ItemType Directory -Force -Path $Output | Out-Null
& dotnet pack $project `
    --configuration Release `
    --output $Output `
    -p:PackageVersion=$Version `
    -p:ContinuousIntegrationBuild=true
if ($LASTEXITCODE -ne 0) {
    throw "dotnet pack failed."
}

$packageName = "GenDigital.OnnxRuntimeHal.Source.$Version.nupkg"
$packagePath = Join-Path $Output $packageName
if (-not (Test-Path -LiteralPath $packagePath)) {
    throw "Expected package was not produced: $packagePath"
}

$sha256 = (Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$packagePath.sha256", "$sha256  $packageName`n", $utf8)

$provenance = [ordered]@{
    schema_version = 1
    package_id = "GenDigital.OnnxRuntimeHal.Source"
    package_version = $Version
    package_file = $packageName
    sha256 = $sha256
    git_commit = $commit
    git_commit_time = ([DateTimeOffset]::Parse($commitTime)).ToUniversalTime().ToString("o")
    source_repository = "https://github.com/odobias/onnx-runtime-hal"
    dirty_source = $dirty
}
$provenancePath = "$packagePath.provenance.json"
[IO.File]::WriteAllText($provenancePath, "$(($provenance | ConvertTo-Json -Depth 4))`n", $utf8)

Write-Host "Packaged $packageName" -ForegroundColor Green
Write-Host "  SHA-256: $sha256"
Write-Host "  Provenance: $provenancePath"
