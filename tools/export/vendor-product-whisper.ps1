# Vendors a product monorepo's framework/whisper into this repo as a self-contained
# Sherpa-export decode stack for the onnx-sherpa benchmark backend.
#
#   $env:PRODUCT_MONOREPO_ROOT = "D:\src\product"
#   .\tools\export\vendor-product-whisper.ps1
#   .\tools\export\vendor-product-whisper.ps1 -ProductRoot D:\src\product

[CmdletBinding()]
param(
    [string]$ProductRoot = $env:PRODUCT_MONOREPO_ROOT
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

if (-not $ProductRoot) {
    throw "Pass -ProductRoot or set PRODUCT_MONOREPO_ROOT."
}

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$srcWhisper = Join-Path $ProductRoot "framework\whisper\src\whisper"
$srcInclude = Join-Path $ProductRoot "framework\whisper\include\asw\framework\whisper"
$dst = Join-Path $root "src\workloads\whisper\sherpa_export"
$dstInc = Join-Path $dst "include\asw\framework\whisper"

if (-not (Test-Path (Join-Path $srcWhisper "Session.cpp"))) {
    throw "Whisper sources not found under $srcWhisper"
}

New-Item -ItemType Directory -Force -Path $dst, $dstInc | Out-Null

$cppFiles = @(
    "Base64.cpp", "Base64.h",
    "BpeTokenizer.cpp", "BpeTokenizer.h",
    "FeatureExtractor.cpp", "FeatureExtractor.h",
    "Fft.cpp", "Fft.h",
    "GreedyDecoder.cpp", "GreedyDecoder.h",
    "LanguageDetector.cpp", "LanguageDetector.h",
    "MelFilters.cpp", "MelFilters.h",
    "Metadata.cpp", "Metadata.h",
    "Resampler.cpp", "Resampler.h",
    "Session.cpp"
)
foreach ($f in $cppFiles) {
    Copy-Item -Force (Join-Path $srcWhisper $f) (Join-Path $dst $f)
}

foreach ($f in @("metadata.h", "options.h", "result.h", "session.h")) {
    Copy-Item -Force (Join-Path $srcInclude $f) (Join-Path $dstInc $f)
}

# Umbrella header
@'
#pragma once
#include <asw/framework/whisper/metadata.h>
#include <asw/framework/whisper/options.h>
#include <asw/framework/whisper/result.h>
#include <asw/framework/whisper/session.h>
'@ | Set-Content -Path (Join-Path $dst "include\asw\framework\whisper.h") -Encoding utf8

# Minimal stdafx that avoids product-framework dependencies.
@'
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
'@ | Set-Content -Path (Join-Path $dst "stdafx.h") -Encoding utf8

# Rewrite includes: onnxruntime/onnxruntime_cxx_api.h -> onnxruntime_cxx_api.h
# and public headers that pull the onnxruntime/ path.
Get-ChildItem $dst -Recurse -Include *.h,*.hpp,*.cpp | ForEach-Object {
    $text = [System.IO.File]::ReadAllText($_.FullName)
    $new = $text.Replace('<onnxruntime/onnxruntime_cxx_api.h>', '<onnxruntime_cxx_api.h>')
    if ($new -ne $text) {
        [System.IO.File]::WriteAllText($_.FullName, $new)
    }
}

# NOTICE for provenance
@"
Vendored from the consumer product's framework/whisper component.
Source component: framework/whisper
Do not hand-edit for product fixes — re-run tools/export/vendor-product-whisper.ps1.
"@ | Set-Content -Path (Join-Path $dst "VENDOR.md") -Encoding utf8

Write-Host "Vendored product whisper into $dst" -ForegroundColor Green
