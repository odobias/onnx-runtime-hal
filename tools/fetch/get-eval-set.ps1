# Builds a small labeled eval set (WAV + reference transcripts) for WER/CER scoring.
# Uses a tiny public LibriSpeech sample via make_eval_set.py in the shared venv.
# Falls back to the single jfk.wav clip (with a known reference) if the download
# fails, so the benchmark harness always has something to run.
#
#   .\get-eval-set.ps1              # up to 20 utterances -> workloads/eval/eval.jsonl
#   .\get-eval-set.ps1 -Count 5

[CmdletBinding()]
param([int]$Count = 20)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$evalDir = Join-Path $root "workloads\eval"
New-Item -ItemType Directory -Force -Path $evalDir | Out-Null

$venv = Join-Path $root ".venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $py) { Write-Host "Python not found. Run bootstrap.ps1 first." -ForegroundColor Red; exit 1 }
    & $py -m venv $venv
}

Write-Host "Ensuring eval-set toolchain (datasets, soundfile)..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install --quiet "datasets" "soundfile" "numpy"

$script = Join-Path $PSScriptRoot "make_eval_set.py"
Write-Host "Building eval set (up to $Count utterances)..." -ForegroundColor Cyan
& $vpy $script --outdir $evalDir --count $Count
$ok = ($LASTEXITCODE -eq 0)

if (-not $ok) {
    Write-Host "Dataset build failed; falling back to jfk.wav clip." -ForegroundColor Yellow
    $jfk = Join-Path $root "workloads\audio\jfk.wav"
    if (-not (Test-Path $jfk)) { & (Join-Path $PSScriptRoot "get-audio.ps1") }
    $ref = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
    $rec = [ordered]@{ id = "jfk"; audio = "workloads/audio/jfk.wav"; ref = $ref; duration_s = 11.0 }
    ($rec | ConvertTo-Json -Compress) | Set-Content -Path (Join-Path $evalDir "eval.jsonl") -Encoding UTF8
    Write-Host "Wrote 1 utterance (jfk) to workloads\eval\eval.jsonl" -ForegroundColor Green
}

exit 0
