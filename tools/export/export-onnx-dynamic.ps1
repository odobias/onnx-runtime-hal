# Exports whisper-tiny(.en) to the STOCK optimum "with-past" ONNX (encoder +
# decoder + decoder_with_past) into src/workloads/whisper/models/dynamic-onnx. This is the model
# the onnx-dynamic backend runs: a real KV cache (encoder cross-attention K/V
# computed once, decoder self-attention K/V grow per token), which is faster than
# the static no-KV recompute on CPU/GPU but is NOT NPU-compilable (dynamic shapes).
#
# Output is gitignored (models/ is never committed); a fresh clone reproduces it
# here or pulls it from HF via get-models.ps1. The intel-onnx (OpenVINO) backend
# also reads this dir (it just ignores decoder_with_past), so the two share it.
#
#   .\tools\export\export-onnx-dynamic.ps1
#   .\tools\export\export-onnx-dynamic.ps1 -Model openai/whisper-tiny -Out whisper-tiny-onnx

[CmdletBinding()]
param(
    [string]$Model = "openai/whisper-tiny.en",
    [string]$Out = "whisper/en-onnx"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$outDir = Join-Path $root "models\$Out"
if (Test-Path (Join-Path $outDir "decoder_with_past_model.onnx")) {
    Write-Host "Already exported: $outDir" -ForegroundColor Green
    return
}

$venv = Join-Path $root ".venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $py) { Write-Host "Python not found. Run bootstrap.ps1 first." -ForegroundColor Red; exit 1 }
    Write-Host "Creating venv at $venv" -ForegroundColor Cyan
    & $py -m venv $venv
}
Write-Host "Ensuring export toolchain (optimum, onnx, onnxruntime)..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install --quiet "optimum[exporters]" onnx onnxruntime

Write-Host "Exporting $Model (with-past) -> $outDir" -ForegroundColor Cyan
$optimum = Join-Path $venv "Scripts\optimum-cli.exe"
& $optimum export onnx --model $Model --task automatic-speech-recognition-with-past $outDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "Export FAILED." -ForegroundColor Red
    if ((& $vpy --version) -match "3\.1[4-9]") {
        Write-Host ("  Hint: on Python 3.14+, huggingface/optimum has a known descriptor bug. In " +
            ".venv\Lib\site-packages\optimum\exporters\base.py, change " +
            "`"self.NORMALIZED_CONFIG_CLASS(self._config)`" to " +
            "`"type(self).NORMALIZED_CONFIG_CLASS(self._config)`". Or use Python <=3.13.") -ForegroundColor Yellow
    }
    exit 1
}

# The exporter also emits decoder_model_merged.onnx (a use_cache_branch model that
# folds the no-past + with-past decoders). The onnx-dynamic backend uses the two
# separate files, so drop the merged one to keep the package (and reported model
# size) lean.
$merged = Join-Path $outDir "decoder_model_merged.onnx"
if (Test-Path $merged) { Remove-Item $merged -Force; Write-Host "Removed unused decoder_model_merged.onnx" -ForegroundColor DarkGray }

$sizeMb = [math]::Round(((Get-ChildItem $outDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "Dynamic ONNX ready: $outDir ($sizeMb MB)" -ForegroundColor Green
