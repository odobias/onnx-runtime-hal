# Fetches a C++ ONNX Runtime SDK (headers + import lib + runtime DLLs) from the
# public NuGet feed into third_party/onnxruntime/, which is where backend.ort.props
# auto-detects it (OrtDir). This is what lets the ORT-based C++ backends (and the
# deepfake classifier harness) build on a box that only has the *Python* onnxruntime
# wheel -- the wheel ships onnxruntime.dll but NO C++ headers or import lib.
#
# Default package is the DirectML flavour (CPU + DirectML GPU EPs, x64), matching the
# onnxruntime-directml Python wheel used by the experiments. Pick the version to line
# up with whatever `python -c "import onnxruntime; print(onnxruntime.__version__)"`
# reports so the C++ and Python paths measure the same runtime build.
#
#   .\scripts\get-ort-sdk.ps1                       # DirectML, 1.24.4, x64
#   .\scripts\get-ort-sdk.ps1 -Version 1.24.4
#   .\scripts\get-ort-sdk.ps1 -Package Microsoft.ML.OnnxRuntime -Version 1.24.4   # CPU-only

[CmdletBinding()]
param(
    [string]$Package = "Microsoft.ML.OnnxRuntime.DirectML",
    [string]$Version = "1.24.4",
    [ValidateSet("win-x64", "win-arm64")][string]$Rid = "win-x64",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
if (-not $OutDir) { $OutDir = Join-Path (Join-Path $root "third_party") "onnxruntime" }

$pkgLower = $Package.ToLowerInvariant()
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ortsdk_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$nupkg = Join-Path $tmp "$pkgLower.$Version.nupkg"
$uri = "https://api.nuget.org/v3-flatcontainer/$pkgLower/$Version/$pkgLower.$Version.nupkg"

Write-Host "Downloading $uri" -ForegroundColor Cyan
$ProgressPreference = "SilentlyContinue"
Invoke-WebRequest -Uri $uri -OutFile $nupkg

$extract = Join-Path $tmp "x"
Expand-Archive -Path $nupkg -DestinationPath $extract -Force

# NuGet layout: build/native/include/*.h, runtimes/<rid>/native/{onnxruntime.dll,onnxruntime.lib,DirectML.dll}
$incSrc = Join-Path $extract "build\native\include"
$natSrc = Join-Path $extract "runtimes\$Rid\native"
if (-not (Test-Path $incSrc)) { throw "headers not found in package at $incSrc" }
if (-not (Test-Path $natSrc)) { throw "native runtime not found in package at $natSrc" }

$incDst = Join-Path $OutDir "include"
$libDst = Join-Path $OutDir "lib"
$binDst = Join-Path $OutDir "bin"
foreach ($d in @($incDst, $libDst, $binDst)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

Copy-Item (Join-Path $incSrc "*") $incDst -Recurse -Force
# Import lib lands in lib/ (linker) and DLLs in bin/ (staging), mirroring an SDK tree.
Get-ChildItem $natSrc -Filter *.lib | ForEach-Object { Copy-Item $_.FullName $libDst -Force }
Get-ChildItem $natSrc -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName $binDst -Force
    Copy-Item $_.FullName $libDst -Force   # keep dll beside lib too; some stages look there
}

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "ORT SDK ready: $OutDir" -ForegroundColor Green
Write-Host ("  headers: {0}" -f (Get-ChildItem $incDst -Filter *.h | Measure-Object).Count)
Get-ChildItem $incDst -Filter *provider_factory.h | ForEach-Object { Write-Host ("    {0}" -f $_.Name) -ForegroundColor DarkGray }
Write-Host ("  lib    : {0}" -f ((Get-ChildItem $libDst -Filter *.lib).Name -join ", "))
Write-Host ("  bin    : {0}" -f ((Get-ChildItem $binDst -Filter *.dll).Name -join ", "))
