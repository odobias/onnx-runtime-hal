# Prepares ONNX Runtime + Plugin QNN EP for the Snapdragon/Hexagon backend.
# Uses pip wheels (onnxruntime + onnxruntime-qnn) for runtime DLLs and the
# Microsoft.ML.OnnxRuntime NuGet for C++ headers/libs.
#
#   .\setup-qualcomm.ps1
#   .\setup-qualcomm.ps1 -OrtVersion 1.27.0 -QnnVersion 2.3.0
[CmdletBinding()]
param(
    [string]$OrtVersion = "1.27.0",
    [string]$QnnVersion = "2.3.0"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path $PSScriptRoot -Parent
$thirdParty = Join-Path $root "third_party"
$ortDir = Join-Path $thirdParty "onnxruntime"
$qnnDir = Join-Path $thirdParty "qnn-ep"
New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

function Resolve-Python {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { throw "Python not found. Install Python 3.12+ first." }
    return $py
}

function Resolve-QnnLibArch {
    $machine = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
    if ($machine -eq "AMD64") { return "arm64ec" }
  # Native ARM64 Python on Snapdragon uses arm64ec libs when PROCESSOR_ARCHITECTURE is AMD64
  # (x64-emulated Python). Fall back to arm64ec for Windows ARM64 hosts.
    if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") {
        return "arm64ec"
    }
    return "amd64"
}

Write-Host "Installing ONNX Runtime + Plugin QNN EP (pip)..." -ForegroundColor Cyan
$py = Resolve-Python
& $py -m pip install --upgrade pip --quiet
& $py -m pip install --quiet "onnxruntime==$OrtVersion" "onnxruntime-qnn==$QnnVersion"

$site = (& $py -c "import onnxruntime_qnn, os; print(os.path.dirname(onnxruntime_qnn.__file__))").Trim()
$arch = Resolve-QnnLibArch
$qnnLibs = Join-Path $site "libs\$arch"
if (-not (Test-Path $qnnLibs)) {
    throw "QNN libs not found at $qnnLibs (arch=$arch)"
}

Write-Host "Staging QNN EP runtime from $qnnLibs -> $qnnDir" -ForegroundColor Cyan
if (Test-Path $qnnDir) { Remove-Item $qnnDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $qnnDir | Out-Null
Copy-Item -Path (Join-Path $qnnLibs "*") -Destination $qnnDir -Recurse -Force

$ortSite = (& $py -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))").Trim()
$ortCapi = Join-Path $ortSite "capi"
Copy-Item (Join-Path $ortCapi "onnxruntime.dll") $qnnDir -Force
Copy-Item (Join-Path $ortCapi "onnxruntime_providers_shared.dll") $qnnDir -Force -ErrorAction SilentlyContinue

if (-not (Test-Path (Join-Path $ortDir "include\onnxruntime_cxx_api.h"))) {
    Write-Host "Fetching Microsoft.ML.OnnxRuntime $OrtVersion (NuGet) for C++ headers..." -ForegroundColor Cyan
    $nuget = Join-Path $thirdParty ".nuget\nuget.exe"
    if (-not (Test-Path $nuget)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $nuget) | Out-Null
        Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile $nuget -UseBasicParsing
    }
    $pkgDir = Join-Path $thirdParty ".nuget\packages"
    & $nuget install Microsoft.ML.OnnxRuntime -Version $OrtVersion -OutputDirectory $pkgDir -NonInteractive | Out-Null
    $pkgRoot = Join-Path $pkgDir "Microsoft.ML.OnnxRuntime.$OrtVersion"
    if (Test-Path $ortDir) { Remove-Item $ortDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $ortDir | Out-Null
    Copy-Item (Join-Path $pkgRoot "build\native\include") (Join-Path $ortDir "include") -Recurse -Force
    New-Item -ItemType Directory -Force -Path (Join-Path $ortDir "lib") | Out-Null
    Copy-Item (Join-Path $pkgRoot "runtimes\win-x64\native\onnxruntime.lib") (Join-Path $ortDir "lib") -Force
    New-Item -ItemType Directory -Force -Path (Join-Path $ortDir "bin") | Out-Null
    Copy-Item (Join-Path $pkgRoot "runtimes\win-x64\native\onnxruntime.dll") (Join-Path $ortDir "bin") -Force
}

if (-not [Environment]::GetEnvironmentVariable("QNN_EP_DIR", "User")) {
    [Environment]::SetEnvironmentVariable("QNN_EP_DIR", $qnnDir, "User")
}
if (-not [Environment]::GetEnvironmentVariable("ORT_DIR", "User")) {
    [Environment]::SetEnvironmentVariable("ORT_DIR", $ortDir, "User")
}

Write-Host "Qualcomm toolchain ready:" -ForegroundColor Green
Write-Host "  ORT SDK : $ortDir"
Write-Host "  QNN EP  : $qnnDir (arch=$arch)"
Write-Host "Build with: .\scripts\build.ps1 -EnableQualcomm -DisableIntel"
