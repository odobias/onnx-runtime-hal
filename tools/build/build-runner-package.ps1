[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release",
    [ValidateSet("x64", "ARM64")][string]$Architecture = "",
    [ValidateSet("ort", "dml", "winml", "ovep", "amd", "qualcomm")]
    [string[]]$Runner = @(),
    [switch]$SkipBuild,
    [switch]$SkipAssets,
    [switch]$Clean,
    [string]$Output = "",
    [string[]]$AdditionalRunnerRoot = @(),
    [string]$RyzenAiDir = "",
    [string]$OrtDir = ""
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $Architecture) {
    $Architecture = if (
        [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq
        [System.Runtime.InteropServices.Architecture]::Arm64
    ) { "ARM64" } else { "x64" }
}
if (-not $Output) {
    $Output = Join-Path $root "dist\npu-inference-bench-$Architecture"
}
$Output = [IO.Path]::GetFullPath($Output)
$catalog = Get-Content (Join-Path $root "packaging\runner-catalog.json") `
    -Raw -Encoding UTF8 | ConvertFrom-Json
$definitions = @($catalog.runners)
$selectedIds = if ($Runner.Count) { @($Runner) } else {
    @($definitions | Where-Object { @($_.architectures) -contains $Architecture } |
        ForEach-Object { [string]$_.id })
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-PeArchitecture([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $reader = [IO.BinaryReader]::new($stream)
        if ($reader.ReadUInt16() -ne 0x5a4d) { throw "not a PE executable" }
        $stream.Position = 0x3c
        $peOffset = $reader.ReadInt32()
        if ($peOffset -lt 0 -or $peOffset -gt ($stream.Length - 6)) {
            throw "invalid PE header offset"
        }
        $stream.Position = $peOffset
        if ($reader.ReadUInt32() -ne 0x00004550) { throw "invalid PE signature" }
        switch ($reader.ReadUInt16()) {
            0x8664 { return "x64" }
            0xaa64 { return "ARM64" }
            default { throw "unsupported PE machine" }
        }
    } finally {
        if ($reader) { $reader.Dispose() } else { $stream.Dispose() }
    }
}

function Get-Definition([string]$Id) {
    $matches = @($definitions | Where-Object { [string]$_.id -eq $Id })
    if ($matches.Count -ne 1) { throw "Unknown runner '$Id'." }
    return $matches[0]
}

function Get-RunnerFiles([string]$Source) {
    return @(Get-ChildItem -LiteralPath $Source -File | Where-Object {
        $_.Extension -in @(".exe", ".dll")
    } | Sort-Object Name)
}

function New-FileRecords([string]$Directory, [object[]]$Files) {
    return @($Files | ForEach-Object {
        [ordered]@{
            path = $_.Name
            sha256 = Get-Sha256 $_.FullName
            size_bytes = [int64]$_.Length
        }
    })
}

function New-RunnerRecord([object]$Definition, [string]$Source, [string]$Destination) {
    $files = Get-RunnerFiles $Source
    $names = @($files | ForEach-Object { $_.Name })
    foreach ($required in @($Definition.required_files)) {
        if ($names -notcontains [string]$required) {
            throw "Runner '$($Definition.id)' is missing required file '$required' in $Source."
        }
    }
    $exe = Join-Path $Source "NpuInferenceBench.exe"
    $actualArchitecture = Get-PeArchitecture $exe
    if ($actualArchitecture -ne $Architecture) {
        throw "Runner '$($Definition.id)' targets $actualArchitecture, package targets $Architecture."
    }

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($file in $files) {
        Copy-Item -LiteralPath $file.FullName -Destination $Destination -Force
    }
    $copied = Get-RunnerFiles $Destination
    $record = [ordered]@{
        id = [string]$Definition.id
        runtime_target = [string]$Definition.runtime_target
        architecture = $Architecture
        configuration = $Configuration
        platform_tag = "$Architecture$([string]$Definition.platform_suffix)"
        path = "runners/$($Definition.id)"
        executable = "NpuInferenceBench.exe"
        vendors = @($Definition.vendors)
        provider_patterns = @($Definition.provider_patterns)
        priority = [int]$Definition.priority
        minimum_os_build = if ($Definition.minimum_os_build) {
            [int]$Definition.minimum_os_build
        } else { $null }
        required_files = @($Definition.required_files)
        files = New-FileRecords $Destination $copied
    }
    $record | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath (Join-Path $Destination "runner.json") -Encoding UTF8
    return $record
}

function Invoke-RunnerBuild([object]$Definition) {
    $arguments = @{
        Platform = $Architecture
        Configuration = $Configuration
        DisableIntel = $true
    }
    $arguments[[string]$Definition.build_switch] = $true
    if ($RyzenAiDir) { $arguments.RyzenAiDir = $RyzenAiDir }
    if ($OrtDir) { $arguments.OrtDir = $OrtDir }
    & (Join-Path $root "tools\build\build.ps1") @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed for runner '$($Definition.id)'."
    }
}

function Get-ExternalArtifactDirectories([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    if (Test-Path -LiteralPath (Join-Path $resolved "runner.json")) {
        return @($resolved)
    }
    return @(Get-ChildItem -LiteralPath $resolved -Directory | Where-Object {
        Test-Path -LiteralPath (Join-Path $_.FullName "runner.json")
    } | ForEach-Object { $_.FullName })
}

function Import-RunnerArtifact([string]$Artifact, [hashtable]$Seen) {
    $metadata = Get-Content (Join-Path $Artifact "runner.json") -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $id = [string]$metadata.id
    $definition = Get-Definition $id
    if ($Seen.ContainsKey($id)) { throw "Duplicate runner '$id'." }
    if ([string]$metadata.architecture -ne $Architecture) {
        throw "Runner '$id' targets $($metadata.architecture), package targets $Architecture."
    }
    if (@($definition.architectures) -notcontains $Architecture) {
        throw "Runner '$id' does not support package architecture $Architecture."
    }

    $declared = @($metadata.files)
    $paths = @{}
    foreach ($file in $declared) {
        $relative = ([string]$file.path).Replace('\', '/')
        if (-not $relative -or [IO.Path]::IsPathRooted($relative) -or
            $relative -match '(^|/)\.\.(/|$)' -or $paths.ContainsKey($relative)) {
            throw "Runner '$id' declares unsafe or duplicate file '$relative'."
        }
        $paths[$relative] = $true
        $source = Join-Path $Artifact ($relative -replace '/', '\')
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Runner '$id' declares missing file '$relative'."
        }
        if ($file.sha256 -and (Get-Sha256 $source) -ne ([string]$file.sha256).ToLowerInvariant()) {
            throw "Runner '$id' file '$relative' has the wrong SHA-256."
        }
    }
    if (@($declared | Where-Object {
        ([string]$_.path).Replace('\', '/') -eq "NpuInferenceBench.exe"
    }).Count -ne 1) {
        throw "Runner '$id' must declare exactly one NpuInferenceBench.exe."
    }
    foreach ($required in @($definition.required_files)) {
        if (-not $paths.ContainsKey([string]$required)) {
            throw "Runner '$id' is missing required file '$required'."
        }
    }
    $exeArchitecture = Get-PeArchitecture (Join-Path $Artifact "NpuInferenceBench.exe")
    if ($exeArchitecture -ne $Architecture) {
        throw "Runner '$id' executable targets $exeArchitecture, package targets $Architecture."
    }

    $destination = Join-Path $Output "runners\$id"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($file in $declared) {
        $relative = ([string]$file.path).Replace('/', '\')
        $target = Join-Path $destination $relative
        New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $Artifact $relative) -Destination $target -Force
    }
    $sourceFiles = Get-RunnerFiles $destination
    $record = [ordered]@{
        id = $id
        runtime_target = [string]$definition.runtime_target
        architecture = $Architecture
        configuration = $Configuration
        platform_tag = "$Architecture$([string]$definition.platform_suffix)"
        path = "runners/$id"
        executable = "NpuInferenceBench.exe"
        vendors = @($definition.vendors)
        provider_patterns = @($definition.provider_patterns)
        priority = [int]$definition.priority
        minimum_os_build = if ($definition.minimum_os_build) {
            [int]$definition.minimum_os_build
        } else { $null }
        required_files = @($definition.required_files)
        files = New-FileRecords $destination $sourceFiles
    }
    $record | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath (Join-Path $destination "runner.json") -Encoding UTF8
    $Seen[$id] = $true
    return $record
}

if ($Clean -and (Test-Path -LiteralPath $Output)) {
    Remove-Item -LiteralPath $Output -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $Output "runners") -Force | Out-Null

$records = [System.Collections.Generic.List[object]]::new()
$skipped = [System.Collections.Generic.List[object]]::new()
$seen = @{}
foreach ($id in $selectedIds) {
    $definition = Get-Definition $id
    if (@($definition.architectures) -notcontains $Architecture) {
        $skipped.Add([ordered]@{ id = $id; reason = "unsupported on $Architecture" })
        continue
    }
    $source = Join-Path $root "build\$Architecture$([string]$definition.platform_suffix)\$Configuration"
    if (-not $SkipBuild) { Invoke-RunnerBuild $definition }
    if (-not (Test-Path -LiteralPath (Join-Path $source "NpuInferenceBench.exe"))) {
        $skipped.Add([ordered]@{ id = $id; reason = "build output is missing: $source" })
        continue
    }
    $destination = Join-Path $Output "runners\$id"
    $records.Add((New-RunnerRecord $definition $source $destination))
    $seen[$id] = $true
}

foreach ($additionalRoot in $AdditionalRunnerRoot) {
    foreach ($artifact in @(Get-ExternalArtifactDirectories $additionalRoot)) {
        $records.Add((Import-RunnerArtifact $artifact $seen))
    }
}
$includedIds = @($records | ForEach-Object { [string]$_.id })
$effectiveSkipped = @($skipped | Where-Object { $includedIds -notcontains [string]$_.id })

if (-not $SkipAssets) {
    foreach ($directory in @("benchmark", "models", "tools", "workloads")) {
        $source = Join-Path $root $directory
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination $Output -Recurse -Force
        }
    }
    Copy-Item -LiteralPath (Join-Path $root "run-benchmark.ps1") -Destination $Output -Force
}

$manifest = [ordered]@{
    schema_version = 1
    package_id = "npu-inference-bench-$Architecture"
    architecture = $Architecture
    configuration = $Configuration
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    runners = @($records)
    skipped = $effectiveSkipped
}
$manifest | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath (Join-Path $Output "runner-package.json") -Encoding UTF8
Write-Host "Runner package: $Output" -ForegroundColor Green
foreach ($record in $records) {
    Write-Host "  included: $($record.id)" -ForegroundColor Cyan
}
foreach ($skip in $effectiveSkipped) {
    Write-Host "  skipped : $($skip.id) - $($skip.reason)" -ForegroundColor DarkYellow
}
