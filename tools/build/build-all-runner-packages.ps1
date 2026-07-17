# Build redistributable runner packages for every supported architecture.
#
# Two-phase orchestration:
#   1. Gate / stage every third-party SDK into non-overlapping paths.
#   2. Launch every runner build (all arches) in one parallel pool, then assemble
#      dist/npu-inference-bench-<arch>/ packages.
#
# Vendor ORT packs cannot share one process, so each runner is still a separate
# MSBuild of NpuInferenceBench.sln. OutDir/IntDir are unique per PlatformOutTag,
# and DirectML ORT is staged per architecture (onnxruntime-directml-x64 /
# onnxruntime-directml-ARM64), so the build wave is safe to parallelize after
# staging completes.
#
#   .\tools\build\build-all-runner-packages.ps1
#   .\tools\build\build-all-runner-packages.ps1 -SkipSdkStage          # prereqs already staged
#   .\tools\build\build-all-runner-packages.ps1 -Clean -InstallArm64Tools
#   .\tools\build\build-all-runner-packages.ps1 -ThrottleLimit 6
#
# Outputs:
#   dist/npu-inference-bench-x64/
#   dist/npu-inference-bench-ARM64/
#   dist/runner-packages-summary.json

[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release",
    [ValidateSet("x64", "ARM64")][string[]]$Architecture = @("x64", "ARM64"),
    [ValidateSet("ort", "dml", "winml", "ovep", "amd", "qualcomm")]
    [string[]]$Runner = @(),
    [switch]$SkipBuild,
    [switch]$SkipAssets,
    [switch]$SkipSdkStage,
    [switch]$Clean,
    [switch]$InstallArm64Tools,
    [string]$RyzenAiDir = "",
    [string]$OrtDir = "",
    # Max concurrent build.ps1 processes across the whole matrix. 0 = auto.
    [int]$ThrottleLimit = 0
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$buildDir = Join-Path $root "tools\build"
$setupDir = Join-Path $root "tools\setup"
$fetchDir = Join-Path $root "tools\fetch"
$packageScript = Join-Path $buildDir "build-runner-package.ps1"
$buildScript = Join-Path $buildDir "build.ps1"
$catalogPath = Join-Path $root "packaging\runner-catalog.json"
$vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsSetup = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
$catalog = Get-Content $catalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
$definitions = @($catalog.runners)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Get-DirectMlOrtDir([string]$Architecture) {
    if ($OrtDir) { return $OrtDir }
    return (Join-Path $root "third_party\onnxruntime-directml-$Architecture")
}

function Test-OrtSdkRoot([string]$Path) {
    return (Test-Path -LiteralPath (Join-Path $Path "include\onnxruntime_cxx_api.h")) -and
        (Test-Path -LiteralPath (Join-Path $Path "bin\onnxruntime.dll"))
}

function Test-Arm64Toolchain {
    if (-not (Test-Path $vsw)) { return $false }
    $vs = & $vsw -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 `
        -property installationPath 2>$null
    if (-not $vs) { return $false }
    $msvcRoot = Join-Path $vs "VC\Tools\MSVC"
    if (-not (Test-Path $msvcRoot)) { return $false }
    $hostArm64 = @(Get-ChildItem $msvcRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "bin\Hostx64\arm64\cl.exe" } |
        Where-Object { Test-Path $_ })
    return ($hostArm64.Count -gt 0)
}

function Install-Arm64Toolchain {
    if (-not (Test-Path $vsSetup)) {
        throw "Visual Studio Installer not found at $vsSetup"
    }
    if (-not (Test-Path $vsw)) {
        throw "vswhere.exe not found at $vsw"
    }
    $installPath = & $vsw -latest -products * -property installationPath 2>$null
    if (-not $installPath) {
        throw "No Visual Studio installation found to modify for ARM64 tools."
    }

    Write-Host "Installing MSVC ARM64 tools into: $installPath" -ForegroundColor Yellow
    Write-Host "This requires elevation (UAC). Approve the prompt if shown." -ForegroundColor Yellow
    $argString = @(
        "modify",
        "--installPath `"$installPath`"",
        "--add Microsoft.VisualStudio.Component.VC.Tools.ARM64",
        "--passive",
        "--wait",
        "--norestart"
    ) -join " "
    try {
        $proc = Start-Process -FilePath $vsSetup -ArgumentList $argString `
            -Wait -PassThru -Verb RunAs
    } catch {
        throw @"
Failed to launch elevated VS Installer: $($_.Exception.Message)
Install manually: Visual Studio Installer -> Build Tools -> Individual components
  -> 'MSVC v143 - VS 2022 C++ ARM64 build tools', then re-run without -InstallArm64Tools.
"@
    }
    if ($null -eq $proc) {
        throw "Elevated VS Installer launch was cancelled or produced no process."
    }
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw @"
VS Installer failed installing ARM64 tools (exit $($proc.ExitCode)).
Install manually: Visual Studio Installer -> Build Tools -> Individual components
  -> 'MSVC v143 - VS 2022 C++ ARM64 build tools', then re-run without -InstallArm64Tools.
"@
    }
    if (-not (Test-Arm64Toolchain)) {
        throw "ARM64 toolchain still missing after VS Installer modify. Reboot or install the ARM64 C++ build tools manually."
    }
}

function Invoke-Native([string]$Script, [hashtable]$Arguments) {
    $parts = foreach ($entry in $Arguments.GetEnumerator()) {
        $name = [string]$entry.Key
        $value = $entry.Value
        if ($value -is [bool]) {
            if ($value) { "-$name" }
        } elseif ($null -ne $value -and "$value" -ne "") {
            "-$name $value"
        }
    }
    Write-Host ("-> {0} {1}" -f $Script, ($parts -join ' ')) -ForegroundColor DarkGray
    & $Script @Arguments
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Script"
    }
}

function Stage-AllPrereqs([string[]]$Architectures) {
    if ($SkipSdkStage) {
        Write-Step "SDK staging skipped (-SkipSdkStage) - verifying staged roots"
        foreach ($arch in $Architectures) {
            $dml = Get-DirectMlOrtDir $arch
            if (Test-OrtSdkRoot $dml) {
                Write-Host "  ok DirectML ORT $arch : $dml" -ForegroundColor Green
            } else {
                Write-Host "  missing DirectML ORT $arch : $dml" -ForegroundColor DarkYellow
            }
            $winmlDll = Join-Path $root "third_party\windows-ml\bin\$arch\onnxruntime.dll"
            if (Test-Path -LiteralPath $winmlDll) {
                Write-Host "  ok Windows ML $arch" -ForegroundColor Green
            } else {
                Write-Host "  missing Windows ML $arch : $winmlDll" -ForegroundColor DarkYellow
            }
        }
        if ($Architectures -contains "x64") {
            $ovep = Join-Path $root "third_party\onnxruntime-openvino\bin\onnxruntime_providers_openvino.dll"
            if (Test-Path -LiteralPath $ovep) {
                Write-Host "  ok OVEP" -ForegroundColor Green
            } else {
                Write-Host "  missing OVEP : $ovep" -ForegroundColor DarkYellow
            }
        }
        if ($Architectures -contains "ARM64") {
            $qnn = Join-Path $root "third_party\qnn-ep\onnxruntime_providers_qnn.dll"
            if (Test-Path -LiteralPath $qnn) {
                Write-Host "  ok Qualcomm QNN EP" -ForegroundColor Green
            } else {
                Write-Host "  missing Qualcomm QNN EP : $qnn" -ForegroundColor DarkYellow
            }
        }
        return
    }

    Write-Step "Stage all third-party SDKs (gate before parallel builds)"

    foreach ($arch in $Architectures) {
        $dmlRid = if ($arch -eq "ARM64") { "arm64" } else { "x64" }
        $dmlDest = Get-DirectMlOrtDir $arch
        Write-Host "DirectML ORT ($arch) -> $dmlDest" -ForegroundColor DarkCyan
        Invoke-Native (Join-Path $fetchDir "get-onnxruntime-directml.ps1") @{
            Architecture = $dmlRid
            Destination = $dmlDest
        }

        Write-Host "Windows ML ($arch)" -ForegroundColor DarkCyan
        Invoke-Native (Join-Path $setupDir "setup-winml.ps1") @{
            Platform = $arch
        }
    }

    # Keep the legacy flat path populated for single-arch / ad-hoc builds.
    $hostArch = if (
        [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq
        [System.Runtime.InteropServices.Architecture]::Arm64
    ) { "ARM64" } else { "x64" }
    if ($Architectures -contains $hostArch) {
        $flat = Join-Path $root "third_party\onnxruntime-directml"
        $source = Get-DirectMlOrtDir $hostArch
        if ((Test-OrtSdkRoot $source) -and ($flat -ne $source)) {
            Write-Host "Refresh legacy DirectML ORT link target: $flat" -ForegroundColor DarkGray
            if (Test-Path -LiteralPath $flat) {
                Remove-Item -LiteralPath $flat -Recurse -Force
            }
            Copy-Item -LiteralPath $source -Destination $flat -Recurse -Force
        }
    }

    if ($Architectures -contains "x64") {
        Write-Host "OpenVINO EP (x64)" -ForegroundColor DarkCyan
        Invoke-Native (Join-Path $setupDir "setup-ovep.ps1") @{}
        if (-not $env:RYZEN_AI_INSTALLATION_PATH -and -not $RyzenAiDir) {
            $defaultRyzen = "C:\Program Files\RyzenAI\1.7.1"
            if (Test-Path $defaultRyzen) {
                $env:RYZEN_AI_INSTALLATION_PATH = $defaultRyzen
                Write-Host "RYZEN_AI_INSTALLATION_PATH=$defaultRyzen" -ForegroundColor DarkGray
            } else {
                Write-Host "Ryzen AI SDK not found; amd runner will skip if build output is missing." -ForegroundColor DarkYellow
            }
        }
    }

    if ($Architectures -contains "ARM64") {
        # Writes third_party/onnxruntime (ARM64) + third_party/qnn-ep. Safe after
        # DirectML ORT was staged into arch-specific directories; x64 builds pass
        # -OrtDir explicitly and will not read this tree.
        Write-Host "Qualcomm QNN (ARM64)" -ForegroundColor DarkCyan
        Invoke-Native (Join-Path $setupDir "setup-qualcomm.ps1") @{
            Platform = "ARM64"
        }
    }
}

function Get-BuildJobs([string[]]$Architectures) {
    $jobs = [System.Collections.Generic.List[object]]::new()
    foreach ($arch in $Architectures) {
        $selected = if ($Runner.Count) {
            @($definitions | Where-Object {
                ([string]$_.id) -in $Runner -and @($_.architectures) -contains $arch
            })
        } else {
            @($definitions | Where-Object { @($_.architectures) -contains $arch })
        }
        foreach ($definition in $selected) {
            $id = [string]$definition.id
            $jobOrt = $null
            switch ($id) {
                { $_ -in @("ort", "dml") } { $jobOrt = Get-DirectMlOrtDir $arch }
                "qualcomm" { $jobOrt = Join-Path $root "third_party\onnxruntime" }
            }
            $jobs.Add([pscustomobject]@{
                key = "$arch/$id"
                architecture = $arch
                runner = $id
                build_switch = [string]$definition.build_switch
                ort_dir = $jobOrt
            })
        }
    }
    return @($jobs)
}

function Invoke-MatrixBuildsParallel([object[]]$Jobs, [int]$Limit) {
    $logDir = Join-Path $root "build\logs\runner-package"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $innerCpu = [Math]::Max(1, [int][Math]::Floor([Environment]::ProcessorCount / $Limit))
    Write-Host ("Building {0} runners across architectures with ThrottleLimit={1} (MSBuild /m:{2} each)" -f `
        $Jobs.Count, $Limit, $innerCpu) -ForegroundColor Cyan

    $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
    $pwsh = if ($pwshCmd) { $pwshCmd.Source } else { (Get-Process -Id $PID).Path }
    $queue = [System.Collections.Generic.Queue[object]]::new()
    foreach ($job in $Jobs) { $queue.Enqueue($job) }
    $running = [System.Collections.Generic.List[object]]::new()
    $results = [System.Collections.Generic.List[object]]::new()

    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $Limit -and $queue.Count -gt 0) {
            $job = $queue.Dequeue()
            $logPath = Join-Path $logDir ("{0}-{1}.log" -f $job.architecture, $job.runner)
            $errPath = Join-Path $logDir ("{0}-{1}.err.log" -f $job.architecture, $job.runner)
            $argList = [System.Collections.Generic.List[string]]::new()
            $argList.Add("-NoProfile")
            $argList.Add("-File")
            $argList.Add($buildScript)
            $argList.Add("-Platform"); $argList.Add([string]$job.architecture)
            $argList.Add("-Configuration"); $argList.Add($Configuration)
            $argList.Add("-DisableIntel")
            $argList.Add(("-" + [string]$job.build_switch))
            $argList.Add("-MaxCpuCount"); $argList.Add("$innerCpu")
            if ($RyzenAiDir) {
                $argList.Add("-RyzenAiDir"); $argList.Add($RyzenAiDir)
            }
            if ($job.ort_dir) {
                $argList.Add("-OrtDir"); $argList.Add([string]$job.ort_dir)
            }

            Write-Host ("  start {0}" -f $job.key) -ForegroundColor DarkCyan
            $proc = Start-Process -FilePath $pwsh `
                -ArgumentList @($argList.ToArray()) `
                -PassThru -NoNewWindow `
                -RedirectStandardOutput $logPath `
                -RedirectStandardError $errPath
            $running.Add([pscustomobject]@{
                key = $job.key
                process = $proc
                log = $logPath
            })
        }

        Start-Sleep -Milliseconds 400
        $stillRunning = [System.Collections.Generic.List[object]]::new()
        foreach ($item in $running) {
            if (-not $item.process.HasExited) {
                $stillRunning.Add($item)
                continue
            }
            $code = [int]$item.process.ExitCode
            $results.Add([pscustomobject]@{
                key = $item.key
                exit_code = $code
                log = $item.log
            })
            $color = if ($code -eq 0) { "Green" } else { "Red" }
            Write-Host ("  done  {0}: exit={1} log={2}" -f $item.key, $code, $item.log) `
                -ForegroundColor $color
        }
        $running = $stillRunning
    }

    $failures = @($results | Where-Object { $_.exit_code -ne 0 })
    if ($failures.Count) {
        $names = ($failures | ForEach-Object { $_.key }) -join ", "
        throw "Parallel build failed for: $names"
    }
}

function Assemble-ArchitecturePackage([string]$Architecture) {
    Write-Step "Assemble runner package ($Architecture)"
    $arguments = @{
        Configuration = $Configuration
        Architecture = $Architecture
        SkipBuild = $true
    }
    if ($Clean) { $arguments.Clean = $true }
    if ($SkipAssets) { $arguments.SkipAssets = $true }
    if ($Runner.Count) { $arguments.Runner = $Runner }
    if ($RyzenAiDir) { $arguments.RyzenAiDir = $RyzenAiDir }
    $dmlOrt = Get-DirectMlOrtDir $Architecture
    if (Test-OrtSdkRoot $dmlOrt) { $arguments.OrtDir = $dmlOrt }
    Invoke-Native $packageScript $arguments
}

function Get-PackageSummary([string]$Architecture) {
    $manifestPath = Join-Path $root "dist\npu-inference-bench-$Architecture\runner-package.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        return [ordered]@{
            architecture = $Architecture
            status = "missing"
            package = $null
            included = @()
            skipped = @()
            path = $manifestPath
        }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    return [ordered]@{
        architecture = $Architecture
        status = "ok"
        package = [string]$manifest.package_id
        included = @($manifest.runners | ForEach-Object { [string]$_.id })
        skipped = @($manifest.skipped | ForEach-Object {
            [ordered]@{ id = [string]$_.id; reason = [string]$_.reason }
        })
        path = $manifestPath
        generated_at_utc = [string]$manifest.generated_at_utc
    }
}

# --- main --------------------------------------------------------------------

Write-Host "Root          : $root"
Write-Host "Configuration : $Configuration"
Write-Host "Architectures : $($Architecture -join ', ')"
Write-Host "Clean         : $Clean"
Write-Host "SkipSdkStage  : $SkipSdkStage"
Write-Host "SkipBuild     : $SkipBuild"
Write-Host "Started       : $(Get-Date -Format o)"

$results = [System.Collections.Generic.List[object]]::new()
$failures = [System.Collections.Generic.List[string]]::new()
$readyArchitectures = [System.Collections.Generic.List[string]]::new()

foreach ($arch in $Architecture) {
    try {
        if ($arch -eq "ARM64" -and -not (Test-Arm64Toolchain)) {
            if ($InstallArm64Tools) {
                Write-Step "Install ARM64 MSVC toolchain"
                Install-Arm64Toolchain
            } else {
                throw @'
MSVC ARM64 cross tools not installed. Re-run with -InstallArm64Tools, or install
"MSVC v143 - VS 2022 C++ ARM64 build tools" via the VS Installer.
'@
            }
        }
        $readyArchitectures.Add($arch)
    } catch {
        $msg = "$arch toolchain: $($_.Exception.Message)"
        Write-Host $msg -ForegroundColor Red
        $failures.Add($msg)
        $results.Add([ordered]@{
            architecture = $arch
            status = "failed"
            package = $null
            included = @()
            skipped = @(@{ id = "*"; reason = $_.Exception.Message })
            path = (Join-Path $root "dist\npu-inference-bench-$arch\runner-package.json")
        })
    }
}

try {
    if ($readyArchitectures.Count) {
        Stage-AllPrereqs -Architectures @($readyArchitectures)

        if (-not $SkipBuild) {
            Write-Step "Parallel matrix builds"
            $jobs = Get-BuildJobs -Architectures @($readyArchitectures)
            if (-not $jobs.Count) {
                throw "No runner build jobs selected."
            }
            $limit = $ThrottleLimit
            if ($limit -le 0) {
                $limit = [Math]::Max(1, [Math]::Min(6, [Math]::Min(
                    $jobs.Count,
                    [Environment]::ProcessorCount
                )))
            }
            if ($limit -le 1 -or $jobs.Count -le 1) {
                foreach ($job in $jobs) {
                    $arguments = @{
                        Platform = [string]$job.architecture
                        Configuration = $Configuration
                        DisableIntel = $true
                    }
                    $arguments[[string]$job.build_switch] = $true
                    if ($RyzenAiDir) { $arguments.RyzenAiDir = $RyzenAiDir }
                    if ($job.ort_dir) { $arguments.OrtDir = [string]$job.ort_dir }
                    Write-Host ("  build {0}" -f $job.key) -ForegroundColor DarkCyan
                    Invoke-Native $buildScript $arguments
                }
            } else {
                Invoke-MatrixBuildsParallel -Jobs $jobs -Limit $limit
            }
        }

        foreach ($arch in $readyArchitectures) {
            try {
                Assemble-ArchitecturePackage $arch
                $summary = Get-PackageSummary $arch
                $results.Add($summary)
                if ($summary.status -ne "ok") {
                    $failures.Add("$arch package manifest missing after assemble")
                }
            } catch {
                $msg = "$arch assemble failed: $($_.Exception.Message)"
                Write-Host $msg -ForegroundColor Red
                $failures.Add($msg)
                $results.Add([ordered]@{
                    architecture = $arch
                    status = "failed"
                    package = $null
                    included = @()
                    skipped = @(@{ id = "*"; reason = $_.Exception.Message })
                    path = (Join-Path $root "dist\npu-inference-bench-$arch\runner-package.json")
                })
            }
        }
    }
} catch {
    $msg = "matrix failed: $($_.Exception.Message)"
    Write-Host $msg -ForegroundColor Red
    $failures.Add($msg)
    foreach ($arch in $readyArchitectures) {
        if (@($results | Where-Object { $_.architecture -eq $arch }).Count -eq 0) {
            $results.Add([ordered]@{
                architecture = $arch
                status = "failed"
                package = $null
                included = @()
                skipped = @(@{ id = "*"; reason = $_.Exception.Message })
                path = (Join-Path $root "dist\npu-inference-bench-$arch\runner-package.json")
            })
        }
    }
}

Write-Step "Summary"
$summaryPath = Join-Path $root "dist\runner-packages-summary.json"
$payload = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    configuration = $Configuration
    architectures = @($results)
    failures = @($failures)
}
New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath -Parent) | Out-Null
$payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

foreach ($item in $results) {
    $color = switch ($item.status) {
        "ok" { "Green" }
        "failed" { "Red" }
        default { "DarkYellow" }
    }
    Write-Host ("[{0}] {1}" -f $item.architecture, $item.status) -ForegroundColor $color
    if ($item.included.Count) {
        Write-Host ("  included: {0}" -f ($item.included -join ", ")) -ForegroundColor Cyan
    }
    foreach ($skip in @($item.skipped)) {
        Write-Host ("  skipped : {0} - {1}" -f $skip.id, $skip.reason) -ForegroundColor DarkYellow
    }
    Write-Host ("  manifest: {0}" -f $item.path) -ForegroundColor DarkGray
}
Write-Host "Summary JSON : $summaryPath" -ForegroundColor DarkGray
Write-Host "Finished     : $(Get-Date -Format o)"

if ($failures.Count) {
    Write-Host ""
    Write-Host "$($failures.Count) failure(s)." -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "All requested architectures packaged." -ForegroundColor Green
exit 0
