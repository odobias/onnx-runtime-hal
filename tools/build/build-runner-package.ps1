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
    [string]$OrtDir = "",
    # Concurrent runner builds. 0 = auto (up to 4). 1 = sequential.
    # Safe because OutDir/IntDir are per PlatformOutTag; shared third_party
    # trees are read-only during the build phase (stage SDKs first).
    [int]$ThrottleLimit = 0
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
    $Output = Join-Path $root "artifacts\dist\npu-inference-bench-$Architecture"
}
$Output = [IO.Path]::GetFullPath($Output)
$catalog = Get-Content (Join-Path $root "eng\packaging\runner-catalog.json") `
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

function Invoke-RunnerBuild([object]$Definition, [int]$MaxCpuCount = 0) {
    $arguments = @{
        Platform = $Architecture
        Configuration = $Configuration
        DisableIntel = $true
    }
    $arguments[[string]$Definition.build_switch] = $true
    if ($RyzenAiDir) { $arguments.RyzenAiDir = $RyzenAiDir }
    if ($OrtDir) { $arguments.OrtDir = $OrtDir }
    if ($MaxCpuCount -gt 0) { $arguments.MaxCpuCount = $MaxCpuCount }
    & (Join-Path $root "tools\build\build.ps1") @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed for runner '$($Definition.id)'."
    }
}

function Invoke-RunnerBuildsParallel([object[]]$Definitions, [int]$Limit) {
    $buildScript = Join-Path $root "tools\build\build.ps1"
    $logDir = Join-Path $root "artifacts\build\logs\runner-package"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    # Avoid N concurrent /m:auto builds melting the host.
    $innerCpu = [Math]::Max(1, [int][Math]::Floor([Environment]::ProcessorCount / $Limit))
    Write-Host ("Building {0} runners with ThrottleLimit={1} (MSBuild /m:{2} each)" -f `
        $Definitions.Count, $Limit, $innerCpu) -ForegroundColor Cyan

    $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
    $pwsh = if ($pwshCmd) { $pwshCmd.Source } else { (Get-Process -Id $PID).Path }
    $queue = [System.Collections.Generic.Queue[object]]::new()
    foreach ($definition in $Definitions) { $queue.Enqueue($definition) }
    $running = [System.Collections.Generic.List[object]]::new()
    $results = [System.Collections.Generic.List[object]]::new()

    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $Limit -and $queue.Count -gt 0) {
            $definition = $queue.Dequeue()
            $id = [string]$definition.id
            $logPath = Join-Path $logDir ("{0}-{1}.log" -f $Architecture, $id)
            $argList = [System.Collections.Generic.List[string]]::new()
            $argList.Add("-NoProfile")
            $argList.Add("-File")
            $argList.Add($buildScript)
            $argList.Add("-Platform"); $argList.Add($Architecture)
            $argList.Add("-Configuration"); $argList.Add($Configuration)
            $argList.Add("-DisableIntel")
            $argList.Add(("-" + [string]$definition.build_switch))
            $argList.Add("-MaxCpuCount"); $argList.Add("$innerCpu")
            if ($RyzenAiDir) {
                $argList.Add("-RyzenAiDir"); $argList.Add($RyzenAiDir)
            }
            if ($OrtDir) {
                $argList.Add("-OrtDir"); $argList.Add($OrtDir)
            }

            Write-Host ("  start {0}" -f $id) -ForegroundColor DarkCyan
            $proc = Start-Process -FilePath $pwsh `
                -ArgumentList @($argList.ToArray()) `
                -PassThru -NoNewWindow `
                -RedirectStandardOutput $logPath `
                -RedirectStandardError (Join-Path $logDir ("{0}-{1}.err.log" -f $Architecture, $id))
            $running.Add([pscustomobject]@{
                id = $id
                process = $proc
                log = $logPath
            })
        }

        Start-Sleep -Milliseconds 400
        $stillRunning = [System.Collections.Generic.List[object]]::new()
        foreach ($job in $running) {
            if (-not $job.process.HasExited) {
                $stillRunning.Add($job)
                continue
            }
            $code = [int]$job.process.ExitCode
            $results.Add([pscustomobject]@{
                id = $job.id
                exit_code = $code
                log = $job.log
            })
            $color = if ($code -eq 0) { "Green" } else { "Red" }
            Write-Host ("  done  {0}: exit={1} log={2}" -f $job.id, $code, $job.log) `
                -ForegroundColor $color
        }
        $running = $stillRunning
    }

    $failures = @($results | Where-Object { $_.exit_code -ne 0 })
    if ($failures.Count) {
        $names = ($failures | ForEach-Object { $_.id }) -join ", "
        throw "Parallel build failed for runner(s): $names"
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

function Copy-RedistributableAssets([string]$Root, [string]$Output) {
    $catalogPath = Join-Path $Root "eng\packaging\redistributable-assets.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "Redistributable asset catalog missing: $catalogPath"
    }
    $catalog = Get-Content -LiteralPath $catalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $copied = [System.Collections.Generic.List[string]]::new()

    # Prefer artifacts/workloads/classifiers; migrate from legacy models/deepfake when needed.
    $migrate = Join-Path $Root "tools\fetch\migrate-classifier-layout.ps1"
    if (Test-Path -LiteralPath $migrate) {
        & $migrate
    }

    if ($catalog.require_eval_wavs) {
        $evalDir = Join-Path $Root "artifacts\workloads\eval"
        $wavs = @(Get-ChildItem -LiteralPath $evalDir -Filter "ls_*.wav" -File -ErrorAction SilentlyContinue)
        if ($wavs.Count -lt 1) {
            throw "Redistributable eval WAVs missing under artifacts/workloads/eval (ls_*.wav). Run tools/fetch/get-eval-set.ps1 first."
        }
        $evalJsonl = Join-Path $Root "src\workloads\eval\eval.jsonl"
        if (-not (Test-Path -LiteralPath $evalJsonl)) {
            throw "Redistributable eval manifest missing: $evalJsonl"
        }
    }

    function Copy-AssetPath([string]$Relative) {
        $relative = $Relative.Replace('/', '\')
        $source = Join-Path $Root $relative
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Redistributable asset missing: $source"
        }
        $destination = Join-Path $Output $relative
        $destParent = Split-Path $destination -Parent
        if ($destParent) {
            New-Item -ItemType Directory -Path $destParent -Force | Out-Null
        }
        if (Test-Path -LiteralPath $source -PathType Container) {
            Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
        } else {
            Copy-Item -LiteralPath $source -Destination $destination -Force
        }
        $copied.Add(($Relative.Replace('\', '/')))
    }

    foreach ($entry in @($catalog.always_copy)) {
        Copy-AssetPath -Relative ([string]$entry)
    }

    $variants = [System.Collections.Generic.List[object]]::new()
    foreach ($entry in @($catalog.paths)) {
        foreach ($include in @($entry.include)) {
            Copy-AssetPath -Relative ([string]$include)
        }
        $variants.Add([ordered]@{
            id = [string]$entry.id
            workload = [string]$entry.workload
            include = @($entry.include | ForEach-Object { ([string]$_).Replace('\', '/') })
        })
    }

    $slimManifest = Join-Path $Root ([string]$catalog.manifest -replace '/', '\')
    if (-not (Test-Path -LiteralPath $slimManifest)) {
        throw "Redistributable portable manifest missing: $slimManifest"
    }
    $manifestDir = Join-Path $Output "benchmark\manifests"
    New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
    Copy-Item -LiteralPath $slimManifest `
        -Destination (Join-Path $manifestDir "portable.json") -Force
    $copied.Add("benchmark/manifests/portable.json")

    # Packages must not ship a top-level models/ runtime root.
    $modelsOut = Join-Path $Output "models"
    if (Test-Path -LiteralPath $modelsOut) {
        Remove-Item -LiteralPath $modelsOut -Recurse -Force
    }

    $totalBytes = 0L
    $workloadsDir = Join-Path $Output "artifacts\workloads"
    if (Test-Path -LiteralPath $workloadsDir) {
        $totalBytes = @(Get-ChildItem -LiteralPath $workloadsDir -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    }

    Write-Host ("Redistributable assets: {0} variant group(s) under artifacts/workloads/" -f $variants.Count) `
        -ForegroundColor Cyan
    return [ordered]@{
        catalog = "eng/packaging/redistributable-assets.json"
        portable_manifest = "benchmark/manifests/portable.json"
        model_variants = @($variants)
        copied_paths = @($copied)
        total_mb = [math]::Round($totalBytes / 1MB, 1)
    }
}

if ($Clean -and (Test-Path -LiteralPath $Output)) {
    Remove-Item -LiteralPath $Output -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $Output "runners") -Force | Out-Null

$records = [System.Collections.Generic.List[object]]::new()
$skipped = [System.Collections.Generic.List[object]]::new()
$seen = @{}
$buildable = [System.Collections.Generic.List[object]]::new()
foreach ($id in $selectedIds) {
    $definition = Get-Definition $id
    if (@($definition.architectures) -notcontains $Architecture) {
        $skipped.Add([ordered]@{ id = $id; reason = "unsupported on $Architecture" })
        continue
    }
    $buildable.Add($definition)
}

if (-not $SkipBuild -and $buildable.Count) {
    $limit = $ThrottleLimit
    if ($limit -le 0) {
        $limit = [Math]::Max(1, [Math]::Min(4, [Math]::Min(
            $buildable.Count,
            [Environment]::ProcessorCount
        )))
    }
    if ($limit -le 1 -or $buildable.Count -le 1) {
        foreach ($definition in $buildable) {
            Invoke-RunnerBuild $definition
        }
    } else {
        Invoke-RunnerBuildsParallel -Definitions @($buildable) -Limit $limit
    }
}

foreach ($definition in $buildable) {
    $id = [string]$definition.id
    $source = Join-Path $root "artifacts\build\$Architecture$([string]$definition.platform_suffix)\$Configuration"
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

$assetSummary = $null
if (-not $SkipAssets) {
    $assetSummary = Copy-RedistributableAssets -Root $root -Output $Output
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
if ($assetSummary) {
    $manifest.assets = $assetSummary
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
if ($assetSummary) {
    Write-Host ("  assets  : {0} model variant(s), ~{1} MB" -f `
        $assetSummary.model_variants.Count, $assetSummary.total_mb) -ForegroundColor DarkCyan
    foreach ($variant in @($assetSummary.model_variants)) {
        Write-Host ("           - {0} ({1})" -f $variant.id, $variant.workload) -ForegroundColor DarkGray
    }
}
