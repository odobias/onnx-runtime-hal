# Assemble the vendored ONNX Runtime + OpenVINO Execution Provider distribution the
# unified binary links + ships for the Intel native path. Produces:
#
#   artifacts/third_party/onnxruntime-openvino/
#     include/            ORT C/C++ headers            (from the official ORT release)
#     lib/onnxruntime.lib import library               (from the official ORT release)
#     bin/                onnxruntime*.dll (OVEP build) + openvino*/tbb*.dll runtime
#
# Why a matched set: onnxruntime-openvino 1.24.1's provider DLL is built against a
# SPECIFIC OpenVINO ABI (2025.4.1). A mismatched openvino.dll makes OVEP fail to load
# (Win32 error 127, "procedure not found") and ORT silently falls back to CPU. The
# OpenVINO runtime DLLs must also sit NEXT TO the provider DLL -- ORT's provider
# bridge does not honor PATH / AddDllDirectory for that dependency load. build.ps1
# -EnableOvep points OrtDir here and the App vcxproj colocates bin/* beside the exe.
#
# Idempotent: skips assembly when the distro already looks complete (use -Force to
# rebuild). Requires python on PATH (creates artifacts/venv-ovep) and network on first run.
[CmdletBinding()]
param(
    [string]$OrtVersion = "1.24.1",
    [string]$OpenvinoVersion = "2025.4.1",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$tp = Join-Path $root "artifacts\third_party\onnxruntime-openvino"
$venv = Join-Path $root "artifacts/venv-ovep"
$venvPy = Join-Path $venv "Scripts\python.exe"

$lib = Join-Path $tp "lib\onnxruntime.lib"
$ovepDll = Join-Path $tp "bin\onnxruntime_providers_openvino.dll"
$ovDll = Join-Path $tp "bin\openvino.dll"
if (-not $Force -and (Test-Path $lib) -and (Test-Path $ovepDll) -and (Test-Path $ovDll)) {
    Write-Host "OVEP distro already present: $tp (use -Force to rebuild)" -ForegroundColor Green
    return
}

# 1) Matched wheels in a dedicated venv (onnxruntime-openvino pins the OpenVINO ABI).
if (-not (Test-Path $venvPy)) {
    Write-Host "- creating venv $venv" -ForegroundColor DarkCyan
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venv
    } else {
        # Plain python.exe does not accept the py-launcher "-3" switch.
        & python -m venv $venv
    }
    if (-not (Test-Path $venvPy)) { throw "failed to create venv at $venv (is python installed?)" }
}
Write-Host "- installing onnxruntime-openvino==$OrtVersion + openvino==$OpenvinoVersion" -ForegroundColor DarkCyan
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet "onnxruntime-openvino==$OrtVersion" "openvino==$OpenvinoVersion"
if ($LASTEXITCODE -ne 0) { throw "pip install of matched OVEP wheels failed" }

$capi = Join-Path $venv "Lib\site-packages\onnxruntime\capi"
$ovlibs = Join-Path $venv "Lib\site-packages\openvino\libs"
foreach ($d in @($capi, $ovlibs)) {
    if (-not (Test-Path $d)) { throw "expected wheel dir missing: $d" }
}

# 2) Headers + import lib from the official ORT release (the wheel ships neither).
#    The base C API exports match the OVEP-enabled DLL, so this import lib links
#    fine against the runtime DLL we ship in bin/.
$zip = Join-Path $env:TEMP "onnxruntime-win-x64-$OrtVersion.zip"
if (-not (Test-Path $zip)) {
    $url = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-$OrtVersion.zip"
    Write-Host "- downloading ORT release headers+lib: $url" -ForegroundColor DarkCyan
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
}
$ex = Join-Path $env:TEMP "onnxruntime-win-x64-$OrtVersion"
if (Test-Path $ex) { Remove-Item $ex -Recurse -Force }
Expand-Archive $zip -DestinationPath $ex -Force
$relRoot = Get-ChildItem $ex -Directory | Select-Object -First 1 -ExpandProperty FullName

# 3) Assemble the distro.
Write-Host "- assembling $tp" -ForegroundColor DarkCyan
if (Test-Path $tp) { Remove-Item $tp -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $tp "include"), (Join-Path $tp "lib"), (Join-Path $tp "bin") | Out-Null
Copy-Item (Join-Path $relRoot "include\*") (Join-Path $tp "include") -Recurse -Force
Copy-Item (Join-Path $relRoot "lib\onnxruntime.lib") (Join-Path $tp "lib") -Force
# OVEP-enabled ORT DLLs from the wheel + the matched OpenVINO runtime (+ tbb + any
# plugin catalog). Everything lands in one bin/ so the exe can colocate it.
Copy-Item (Join-Path $capi "onnxruntime.dll") (Join-Path $tp "bin") -Force
Copy-Item (Join-Path $capi "onnxruntime_providers_openvino.dll") (Join-Path $tp "bin") -Force
Copy-Item (Join-Path $capi "onnxruntime_providers_shared.dll") (Join-Path $tp "bin") -Force
Copy-Item (Join-Path $ovlibs "*") (Join-Path $tp "bin") -Recurse -Force

$dllCount = (Get-ChildItem (Join-Path $tp "bin") -Filter *.dll).Count
$hdrCount = (Get-ChildItem (Join-Path $tp "include") -Filter *.h).Count
([ordered]@{
    schema_version = 1
    vendor = "Intel"
    package = "onnxruntime-openvino"
    onnxruntime_version = $OrtVersion
    openvino_version = $OpenvinoVersion
    source = "pinned Python wheels plus official ONNX Runtime headers/import library"
} | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath (Join-Path $tp "VERSION.json") -Encoding UTF8
Write-Host ("OVEP distro ready: {0} headers, import lib, {1} runtime DLLs -> {2}" -f $hdrCount, $dllCount, $tp) -ForegroundColor Green
Write-Host "Build the unified Intel binary with:  .\tools\build\build.ps1 -EnableOvep" -ForegroundColor Cyan
