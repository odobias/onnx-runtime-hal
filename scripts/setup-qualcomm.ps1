# Prepares the Qualcomm QNN toolchain for the (scaffold) Snapdragon/Hexagon backend.
# The Qualcomm AI Engine Direct SDK (QNN) is distributed behind a Qualcomm account
# login, so unlike Intel/AMD there is no public direct-download URL to script. This
# script therefore *links* an already-installed QNN SDK into third_party\qnn (the
# same reuse strategy as setup-intel.ps1) and otherwise prints exactly what to fetch.
#
#   .\setup-qualcomm.ps1
#   .\setup-qualcomm.ps1 -QnnSdk "C:\Qualcomm\AIStack\QAIRT\2.x.x.x"
#
# Status: the qualcomm backend is a compile-time throwing stub until the QNN EP
# integration lands (see src/backends/qualcomm/). This just makes the SDK discoverable.
[CmdletBinding()]
param(
    [string]$QnnSdk = $env:QNN_SDK_ROOT
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
$thirdParty = Join-Path $root "third_party"
$target = Join-Path $thirdParty "qnn"
New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

if (Test-Path (Join-Path $target "bin")) {
    Write-Host "QNN SDK already linked at $target" -ForegroundColor Green
    return
}

if ($QnnSdk -and (Test-Path (Join-Path $QnnSdk "bin"))) {
    Write-Host "Linking QNN SDK: $QnnSdk -> $target" -ForegroundColor Cyan
    New-Item -ItemType Junction -Path $target -Target $QnnSdk | Out-Null
    if (-not [Environment]::GetEnvironmentVariable('QNN_SDK_ROOT', 'User')) {
        [Environment]::SetEnvironmentVariable('QNN_SDK_ROOT', $QnnSdk, 'User')
    }
    Write-Host "QNN SDK ready at $target" -ForegroundColor Green
    return
}

Write-Host "Qualcomm QNN SDK not found (scaffold backend)." -ForegroundColor Yellow
Write-Host "To wire it up:" -ForegroundColor Yellow
Write-Host "  1. Download the 'Qualcomm AI Engine Direct SDK' (QAIRT/QNN) from" -ForegroundColor Gray
Write-Host "     https://qpm.qualcomm.com  (requires a Qualcomm account)." -ForegroundColor Gray
Write-Host "  2. Extract it, then re-run:" -ForegroundColor Gray
Write-Host "       .\scripts\setup-qualcomm.ps1 -QnnSdk `"C:\path\to\QAIRT\<version>`"" -ForegroundColor Gray
Write-Host "  3. Runtime uses ONNX Runtime + the QNN EP; the backend stays a throwing" -ForegroundColor Gray
Write-Host "     stub until src/backends/qualcomm is implemented." -ForegroundColor Gray
