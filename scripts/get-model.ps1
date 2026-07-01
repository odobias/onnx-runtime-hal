# Exports whisper-tiny(.en) to OpenVINO IR (GenAI-compatible) into models/.
# Self-contained: creates an isolated Python venv and uses optimum-cli. The output
# is gitignored, so a fresh clone reproduces the model instead of committing blobs.
#
#   .\get-model.ps1                       # openai/whisper-tiny.en -> models\whisper-tiny-en-ov
#   .\get-model.ps1 -Model openai/whisper-tiny -Out whisper-tiny-ov
#   .\get-model.ps1 -WeightFormat int8    # NNCF weight-only INT8

[CmdletBinding()]
param(
    [string]$Model = "openai/whisper-tiny.en",
    [string]$Out = "whisper-tiny-en-ov",
    [ValidateSet("fp32", "fp16", "int8")][string]$WeightFormat = "fp32"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path $PSScriptRoot -Parent
$models = Join-Path $root "models"
$outDir = Join-Path $models $Out
New-Item -ItemType Directory -Force -Path $models | Out-Null

if (Test-Path (Join-Path $outDir "openvino_encoder_model.xml")) {
    Write-Host "Model already exported: $outDir" -ForegroundColor Green
    return
}

# Resolve a Python interpreter.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $py) { Write-Host "Python not found. Install it (winget install Python.Python.3.12) or run bootstrap.ps1." -ForegroundColor Red; exit 1 }

$venv = Join-Path $root ".venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "Creating venv at $venv" -ForegroundColor Cyan
    & $py -m venv $venv
}

Write-Host "Installing export toolchain (optimum-intel[openvino], tokenizers)..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install --quiet "optimum-intel[openvino]" openvino-tokenizers nncf

Write-Host "Exporting $Model ($WeightFormat) -> $outDir" -ForegroundColor Cyan
$optimum = Join-Path $venv "Scripts\optimum-cli.exe"
& $optimum export openvino --model $Model --weight-format $WeightFormat --task automatic-speech-recognition-with-past $outDir
if ($LASTEXITCODE -ne 0) { Write-Host "Export failed." -ForegroundColor Red; exit 1 }

Write-Host "Model ready: $outDir" -ForegroundColor Green
