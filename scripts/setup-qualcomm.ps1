# Prepares ONNX Runtime + Plugin QNN EP for the Snapdragon/Hexagon backend.
#
# Everything is sourced from NuGet so the arch matches the target exactly:
#   - Base ONNX Runtime (headers + import lib + onnxruntime.dll) from
#     Microsoft.ML.OnnxRuntime (runtimes/<rid>/native).
#   - The Plugin QNN EP + Qnn* runtime from Qualcomm.ML.OnnxRuntime.QNN
#     (runtimes/win-arm64/native) -- these are *pure ARM64* (0xAA64), so a native
#     ARM64 exe can load them. (The pip onnxruntime-qnn wheel ships ARM64EC libs,
#     which a pure ARM64 process cannot load -- that path is only for x64.)
#
#   .\setup-qualcomm.ps1
#   .\setup-qualcomm.ps1 -OrtVersion 1.27.0 -QnnVersion 2.3.0
#   .\setup-qualcomm.ps1 -Platform ARM64
[CmdletBinding()]
param(
    [string]$OrtVersion = "1.27.0",
    [string]$QnnVersion = "2.3.0",
    [ValidateSet("x64", "ARM64")][string]$Platform = ""
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

# Default to the host architecture so a native ARM64 host stages win-arm64.
if (-not $Platform) {
    $Platform = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { "ARM64" } else { "x64" }
}
$nugetRid = if ($Platform -eq "ARM64") { "win-arm64" } else { "win-x64" }

$root = Split-Path $PSScriptRoot -Parent
$thirdParty = Join-Path $root "third_party"
$ortDir = Join-Path $thirdParty "onnxruntime"
$qnnDir = Join-Path $thirdParty "qnn-ep"
New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

$nuget = Join-Path $thirdParty ".nuget\nuget.exe"
if (-not (Test-Path $nuget)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $nuget) | Out-Null
    Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile $nuget -UseBasicParsing
}
$pkgDir = Join-Path $thirdParty ".nuget\packages"

function Install-NuGetPackage([string]$Id, [string]$Version) {
    & $nuget install $Id -Version $Version -OutputDirectory $pkgDir -NonInteractive | Out-Null
    $pkgRoot = Join-Path $pkgDir "$Id.$Version"
    if (-not (Test-Path $pkgRoot)) { throw "NuGet package not found after install: $pkgRoot" }
    return $pkgRoot
}

# --- Base ONNX Runtime (headers + import lib + runtime DLLs) -------------------
Write-Host "Fetching Microsoft.ML.OnnxRuntime $OrtVersion (NuGet, $nugetRid)..." -ForegroundColor Cyan
$ortPkg = Install-NuGetPackage "Microsoft.ML.OnnxRuntime" $OrtVersion
$ortNative = Join-Path $ortPkg "runtimes\$nugetRid\native"
if (-not (Test-Path (Join-Path $ortNative "onnxruntime.dll"))) {
    throw "onnxruntime.dll not found in $ortNative (does this package ship $nugetRid?)"
}
if (Test-Path $ortDir) { Remove-Item $ortDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $ortDir "lib"), (Join-Path $ortDir "bin") | Out-Null
Copy-Item (Join-Path $ortPkg "build\native\include") (Join-Path $ortDir "include") -Recurse -Force
Copy-Item (Join-Path $ortNative "onnxruntime.lib") (Join-Path $ortDir "lib") -Force
Copy-Item (Join-Path $ortNative "*.dll") (Join-Path $ortDir "bin") -Force

# --- Plugin QNN EP + Qnn* runtime ---------------------------------------------
if (Test-Path $qnnDir) { Remove-Item $qnnDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $qnnDir | Out-Null

if ($Platform -eq "ARM64") {
    Write-Host "Fetching Qualcomm.ML.OnnxRuntime.QNN $QnnVersion (NuGet, pure ARM64)..." -ForegroundColor Cyan
    $qnnPkg = Install-NuGetPackage "Qualcomm.ML.OnnxRuntime.QNN" $QnnVersion
    $qnnNative = Join-Path $qnnPkg "runtimes\win-arm64\native"
    if (-not (Test-Path (Join-Path $qnnNative "onnxruntime_providers_qnn.dll"))) {
        throw "QNN plugin not found in $qnnNative"
    }
    Copy-Item -Path (Join-Path $qnnNative "*") -Destination $qnnDir -Recurse -Force
    $qnnSource = "$qnnNative (NuGet, pure ARM64)"
}
else {
    # x64 (incl. x64-emulated on a Snapdragon): the pip wheel ships ARM64EC libs
    # for ARM64 hosts (x64-ABI compatible) and amd64 for real x64.
    Write-Host "Installing onnxruntime-qnn $QnnVersion (pip) for x64..." -ForegroundColor Cyan
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { throw "Python not found. Install Python 3.12+ first (x64 QNN path uses the pip wheel)." }
    & $py -m pip install --quiet "onnxruntime-qnn==$QnnVersion"
    $site = (& $py -c "import onnxruntime_qnn, os; print(os.path.dirname(onnxruntime_qnn.__file__))").Trim()
    $arch = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { "arm64ec" } else { "amd64" }
    $qnnLibs = Join-Path $site "libs\$arch"
    if (-not (Test-Path $qnnLibs)) { throw "QNN libs not found at $qnnLibs (arch=$arch)" }
    Copy-Item -Path (Join-Path $qnnLibs "*") -Destination $qnnDir -Recurse -Force
    $qnnSource = "$qnnLibs (pip, arch=$arch)"
}

if (-not [Environment]::GetEnvironmentVariable("QNN_EP_DIR", "User")) {
    [Environment]::SetEnvironmentVariable("QNN_EP_DIR", $qnnDir, "User")
}
if (-not [Environment]::GetEnvironmentVariable("ORT_DIR", "User")) {
    [Environment]::SetEnvironmentVariable("ORT_DIR", $ortDir, "User")
}

Write-Host "Qualcomm toolchain ready:" -ForegroundColor Green
Write-Host "  Platform: $Platform (NuGet rid=$nugetRid)"
Write-Host "  ORT SDK : $ortDir"
Write-Host "  QNN EP  : $qnnDir"
Write-Host "  QNN src : $qnnSource"
Write-Host "Build with: .\scripts\build.ps1 -Platform $Platform -EnableQualcomm -DisableIntel"
