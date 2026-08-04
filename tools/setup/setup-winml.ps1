# Stages the self-contained native Windows ML SDK used by the isolated
# <platform>-winml benchmark build. The package supplies ONNX Runtime, DirectML
# and the flat C Execution Provider Catalog API; vendor EPs are acquired through
# Windows ML at runtime.
#
#   .\tools\setup\setup-winml.ps1
#   .\tools\setup\setup-winml.ps1 -Platform ARM64 -Version 2.1.70
[CmdletBinding()]
param(
    [string]$Version = "2.1.70",
    [ValidateSet("x64", "ARM64")][string]$Platform = ""
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

if (-not $Platform) {
    $Platform = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { "ARM64" } else { "x64" }
}

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$thirdParty = Join-Path $root "artifacts\third_party"
$nuget = Join-Path $thirdParty ".nuget\nuget.exe"
if (-not (Test-Path $nuget)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $nuget) | Out-Null
    Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile $nuget -UseBasicParsing
}

$pkgDir = Join-Path $thirdParty ".nuget\packages"
& $nuget install Microsoft.WindowsAppSDK.ML -Version $Version -OutputDirectory $pkgDir -NonInteractive | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to install Microsoft.WindowsAppSDK.ML $Version" }

# Microsoft.WindowsAppSDK.ML depends on this native, xcopy-deployable package.
$pkg = Join-Path $pkgDir "Microsoft.Windows.AI.MachineLearning.$Version"
if (-not (Test-Path $pkg)) { throw "Windows ML native package not found after restore: $pkg" }

$rid = if ($Platform -eq "ARM64") { "win-arm64" } else { "win-x64" }
$libSource = Join-Path $pkg "lib\native\$Platform"
$binSource = Join-Path $pkg "runtimes\$rid\native"
$includeSource = Join-Path $pkg "include"
foreach ($required in @(
    (Join-Path $includeSource "WinMLEpCatalog.h"),
    (Join-Path $includeSource "winml\onnxruntime_cxx_api.h"),
    (Join-Path $libSource "onnxruntime.lib"),
    (Join-Path $libSource "Microsoft.Windows.AI.MachineLearning.lib"),
    (Join-Path $binSource "onnxruntime.dll"),
    (Join-Path $binSource "Microsoft.Windows.AI.MachineLearning.dll")
)) {
    if (-not (Test-Path $required)) { throw "Windows ML package asset missing: $required" }
}

$staging = Join-Path $thirdParty "windows-ml"
$includeDest = Join-Path $staging "include"
$libDest = Join-Path $staging "lib\$Platform"
$binDest = Join-Path $staging "bin\$Platform"
New-Item -ItemType Directory -Force -Path $includeDest | Out-Null
foreach ($dir in @($libDest, $binDest)) {
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Flatten winml/ so the existing ORT source can keep its standard include names.
Copy-Item (Join-Path $includeSource "winml\*") $includeDest -Recurse -Force
Copy-Item (Join-Path $includeSource "*.h") $includeDest -Force
Copy-Item (Join-Path $libSource "*.lib") $libDest -Force
Copy-Item (Join-Path $binSource "*.dll") $binDest -Force
Set-Content -Path (Join-Path $staging "VERSION") -Value "Microsoft.WindowsAppSDK.ML=$Version`nRID=$rid" -Encoding UTF8
([ordered]@{
    schema_version = 1
    vendor = "Microsoft"
    package = "Microsoft.WindowsAppSDK.ML"
    version = $Version
    rid = $rid
    source = "NuGet package metadata"
} | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath (Join-Path $staging "VERSION.json") -Encoding UTF8

Write-Host "Windows ML $Version staged for $Platform in $staging" -ForegroundColor Green
Write-Host "Build: .\tools\build\build.ps1 -Platform $Platform -EnableWinML -DisableIntel" -ForegroundColor Cyan
