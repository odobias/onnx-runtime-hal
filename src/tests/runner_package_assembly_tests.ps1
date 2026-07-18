[CmdletBinding()]
param([Parameter(Mandatory)][string]$SourceRunner)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$source = (Resolve-Path -LiteralPath $SourceRunner).Path
$sourceMetadata = Get-Content (Join-Path $source "runner.json") -Raw -Encoding UTF8 |
    ConvertFrom-Json
$architecture = [string]$sourceMetadata.architecture
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("npu-package-tests-" + [guid]::NewGuid())
$external = Join-Path $temporaryRoot "external\dml"
$package = Join-Path $temporaryRoot "package"
$buildScript = Join-Path $root "tools\build\build-runner-package.ps1"

function New-DmlArtifact([string]$Destination) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($file in @($sourceMetadata.files)) {
        Copy-Item -LiteralPath (Join-Path $source ([string]$file.path)) `
            -Destination $Destination -Force
    }
    $record = [ordered]@{
        schema_version = 1
        id = "dml"
        runtime_target = "bundled"
        architecture = $architecture
        configuration = "Release"
        platform_tag = "${architecture}-dml"
        path = "runners/dml"
        executable = "NpuInferenceBench.exe"
        vendors = @("Any")
        provider_patterns = @("dml", "directml")
        priority = 40
        minimum_os_build = $null
        files = @($sourceMetadata.files)
    }
    $record | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath (Join-Path $Destination "runner.json") -Encoding UTF8
}

try {
    New-DmlArtifact -Destination $external
    & $buildScript -Architecture $architecture -Runner dml -SkipBuild -SkipAssets `
        -Clean -Output $package -AdditionalRunnerRoot (Split-Path $external -Parent) | Out-Null
    $manifest = Get-Content (Join-Path $package "runner-package.json") -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if (@($manifest.runners).Count -ne 1 -or [string]$manifest.runners[0].id -ne "dml") {
        throw "External runner was not merged into the package."
    }

    $duplicateRejected = $false
    try {
        & $buildScript -Architecture $architecture -Runner dml -SkipBuild -SkipAssets `
            -Clean -Output $package -AdditionalRunnerRoot @(
                (Split-Path $external -Parent),
                (Split-Path $external -Parent)
            ) | Out-Null
    } catch {
        $duplicateRejected = $_.Exception.Message -match "Duplicate runner"
    }
    if (-not $duplicateRejected) { throw "Duplicate external runner ID was not rejected." }

    $metadataPath = Join-Path $external "runner.json"
    $metadata = Get-Content $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $metadata.architecture = if ($architecture -eq "ARM64") { "x64" } else { "ARM64" }
    $metadata | ConvertTo-Json -Depth 10 | Set-Content $metadataPath -Encoding UTF8
    $mixedRejected = $false
    try {
        & $buildScript -Architecture $architecture -Runner dml -SkipBuild -SkipAssets `
            -Clean -Output $package -AdditionalRunnerRoot (Split-Path $external -Parent) | Out-Null
    } catch {
        $mixedRejected = $_.Exception.Message -match "targets"
    }
    if (-not $mixedRejected) { throw "Mixed-architecture artifact was not rejected." }
} finally {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Runner package assembly tests passed."
