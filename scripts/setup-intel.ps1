# Ensures the Intel OpenVINO GenAI C++ SDK is available at third_party\openvino_genai.
# Strategy: reuse an existing local SDK via a directory junction (no re-download);
# otherwise download + extract the archive.
#
#   .\setup-intel.ps1
#   .\setup-intel.ps1 -ExistingSdk "C:\path\to\openvino_genai"
#   .\setup-intel.ps1 -Url <full-zip-url>

[CmdletBinding()]
param(
    [string]$ExistingSdk = "C:\Projects\whisper-npu\cpp\deps\openvino_genai",
    [string]$Version = "2026.2.1.0",
    [string]$Channel = "2026.2.1",
    [string]$Url = ""
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
$thirdParty = Join-Path $root "third_party"
$target = Join-Path $thirdParty "openvino_genai"

New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

if (Test-Path (Join-Path $target "setupvars.ps1")) {
    Write-Host "SDK already present at $target" -ForegroundColor Green
    return
}

# 1) Reuse an existing local SDK via a junction (fast, no download).
if (Test-Path (Join-Path $ExistingSdk "setupvars.ps1")) {
    Write-Host "Linking existing SDK: $ExistingSdk -> $target" -ForegroundColor Cyan
    New-Item -ItemType Junction -Path $target -Target $ExistingSdk | Out-Null
    if (Test-Path (Join-Path $target "setupvars.ps1")) {
        Write-Host "Junction created." -ForegroundColor Green
        return
    }
    Write-Host "Junction failed; falling back to download." -ForegroundColor Yellow
}

# 2) Download + extract the archive.
if ([string]::IsNullOrWhiteSpace($Url)) {
    $Url = "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/$Channel/windows/openvino_genai_windows_${Version}_x86_64.zip"
}
$zip = Join-Path $thirdParty (Split-Path $Url -Leaf)
Write-Host "Downloading $Url" -ForegroundColor Cyan
Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing

Write-Host "Extracting..." -ForegroundColor Cyan
$tmp = Join-Path $thirdParty "_tmp"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive -Path $zip -DestinationPath $tmp -Force
$inner = Get-ChildItem -Directory $tmp | Select-Object -First 1
Move-Item -Path $inner.FullName -Destination $target
Remove-Item -Recurse -Force $tmp
Write-Host "SDK ready at $target" -ForegroundColor Green
