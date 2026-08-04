$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path $PSScriptRoot -Parent) "benchmark\lib\runner-pack.ps1")

$root = Split-Path $PSScriptRoot -Parent
$catalog = Get-Content (Join-Path $root "eng\packaging\runner-catalog.json") -Raw -Encoding UTF8 |
    ConvertFrom-Json
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("npu-runner-tests-" + [guid]::NewGuid())
$failures = [System.Collections.Generic.List[string]]::new()

function Assert-Equal([string]$Name, [object]$Actual, [object]$Expected) {
    if ([string]$Actual -ne [string]$Expected) {
        $failures.Add("$Name`: expected '$Expected', got '$Actual'")
    }
}

function Assert-Throws([string]$Name, [scriptblock]$Action) {
    try {
        & $Action
        $failures.Add("$Name`: expected an exception")
    } catch {}
}

function New-TestPackage([string]$Architecture, [string[]]$Ids) {
    $packageRoot = Join-Path $temporaryRoot ("package-" + $Architecture + "-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null
    $runners = @()
    foreach ($id in $Ids) {
        $definition = @($catalog.runners | Where-Object { $_.id -eq $id })[0]
        $runnerRoot = Join-Path $packageRoot "runners\$id"
        New-Item -ItemType Directory -Path $runnerRoot -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $runnerRoot "NpuInferenceBench.exe") `
            -Value "test" -Encoding Ascii
        $runners += [ordered]@{
            id = $id
            runtime_target = [string]$definition.runtime_target
            architecture = $Architecture
            platform_tag = "$Architecture$([string]$definition.platform_suffix)"
            path = "runners/$id"
            executable = "NpuInferenceBench.exe"
            vendors = @($definition.vendors)
            provider_patterns = @($definition.provider_patterns)
            priority = [int]$definition.priority
            minimum_os_build = if ($definition.minimum_os_build) {
                [int]$definition.minimum_os_build
            } else {
                $null
            }
            files = @()
        }
    }
    $manifest = [ordered]@{
        schema_version = 1
        package_id = "test-$Architecture"
        architecture = $Architecture
        configuration = "Release"
        runners = $runners
        skipped = @()
    }
    $path = Join-Path $packageRoot "runner-package.json"
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $path -Encoding UTF8
    return Get-BenchmarkRunnerPackageManifest -Path $path
}

try {
    $arm = New-TestPackage "ARM64" @("ort", "dml", "winml", "qualcomm")
    Assert-Equal "Qualcomm auto" `
        (Resolve-BenchmarkRunnerPack $arm bundled "" Qualcomm ARM64 99999).id "qualcomm"
    Assert-Equal "ARM64 DirectML" `
        (Resolve-BenchmarkRunnerPack $arm bundled DmlExecutionProvider Qualcomm ARM64 99999).id "dml"
    Assert-Equal "ARM64 Windows ML" `
        (Resolve-BenchmarkRunnerPack $arm winml "" Qualcomm ARM64 99999).id "winml"

    $x64 = New-TestPackage "x64" @("ort", "dml", "winml", "ovep", "amd")
    Assert-Equal "AMD auto" `
        (Resolve-BenchmarkRunnerPack $x64 bundled "" AMD x64 99999).id "amd"
    Assert-Equal "Intel auto" `
        (Resolve-BenchmarkRunnerPack $x64 bundled "" Intel x64 99999).id "ovep"
    Assert-Equal "Intel DirectML" `
        (Resolve-BenchmarkRunnerPack $x64 bundled DmlExecutionProvider Intel x64 99999).id "dml"

    $portable = New-TestPackage "x64" @("ort")
    Assert-Equal "Missing vendor pack fallback" `
        (Resolve-BenchmarkRunnerPack $portable bundled "" AMD x64 99999).id "ort"
    Assert-Throws "Missing explicit OpenVINO" {
        Resolve-BenchmarkRunnerPack $portable bundled OpenVINOExecutionProvider Intel x64 99999
    }
    Assert-Throws "Mixed architecture" {
        Resolve-BenchmarkRunnerPack $x64 bundled "" Intel ARM64 99999
    }
} finally {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}
Write-Host "Runner package resolver tests passed."
