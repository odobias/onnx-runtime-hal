# Copy the HAL static multilingual Whisper package into a consumer MLM tree.
#
#   .\tools\fetch\stage-consumer-whisper-static.ps1 -MlRoot C:\path\to\mlm
#
# Creates:  <MlRoot>\whisper_multilingual\{encoder_model,decoder_model}.onnx + configs

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$MlRoot
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$src = Join-Path $root "artifacts\workloads\whisper\models\static-onnx-tiny-multi-7s"
if (-not (Test-Path (Join-Path $src "encoder_model.onnx"))) {
    throw "Missing $src — run tools/export/export_static_whisper_7s_multi.py or get-models.ps1 first"
}

$dst = Join-Path $MlRoot "whisper_multilingual"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $dst -Force
Write-Host "Staged static multilingual Whisper -> $dst" -ForegroundColor Green
