# One-shot / idempotent copy: models/deepfake -> src/workloads/classifiers
# Preserves the sibling layout fixtures need (../../family/model.onnx).
#
#   .\tools\fetch\migrate-classifier-layout.ps1
#   .\tools\fetch\migrate-classifier-layout.ps1 -Force

[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$src = Join-Path $root "models\deepfake"
$dst = Join-Path $root "src\workloads\classifiers"

if (-not (Test-Path -LiteralPath $src)) {
    Write-Host "No legacy models/deepfake tree; nothing to migrate." -ForegroundColor DarkYellow
    exit 0
}

$sentinel = Join-Path $dst "tsc\model.onnx"
if ((Test-Path -LiteralPath $sentinel) -and -not $Force) {
    Write-Host "src/workloads/classifiers already populated ($sentinel). Pass -Force to overwrite." -ForegroundColor DarkCyan
    exit 0
}

New-Item -ItemType Directory -Force -Path $dst | Out-Null
Write-Host "Migrating $src -> $dst ..." -ForegroundColor Cyan
foreach ($name in @("tsc", "fakeaudio", "audio-samples", "fixtures")) {
    $from = Join-Path $src $name
    if (-not (Test-Path -LiteralPath $from)) { continue }
    $to = Join-Path $dst $name
    if (Test-Path -LiteralPath $to) {
        Remove-Item -LiteralPath $to -Recurse -Force
    }
    Copy-Item -LiteralPath $from -Destination $to -Recurse -Force
    Write-Host "  $name" -ForegroundColor DarkGray
}
Write-Host "Classifier runtime root is now src/workloads/classifiers (models/deepfake is legacy cache)." -ForegroundColor Green
