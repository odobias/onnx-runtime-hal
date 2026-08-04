# Builds a small labeled eval set (WAV + reference transcripts) for WER/CER scoring.
# Uses a tiny public LibriSpeech sample via make_eval_set.py in the shared venv.
# Falls back to the single jfk.wav clip (with a known reference) if the download
# fails, so the benchmark harness always has something to run.
#
#   .\get-eval-set.ps1              # WAVs -> artifacts/workloads/speech; manifest -> src/workloads/eval/eval.jsonl
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
$evalDir = Join-Path $root "artifacts\workloads\speech"
$trackedJsonl = Join-Path $root "src\workloads\eval\eval.jsonl"
New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $trackedJsonl -Parent) | Out-Null

$venv = Join-Path $root "artifacts/venv"
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

$evalPath = $trackedJsonl
$generatedJsonl = Join-Path $evalDir "eval.jsonl"
$priorBaselines = @{}
if (Test-Path -LiteralPath $evalPath) {
    Get-Content -LiteralPath $evalPath -Encoding UTF8 | ForEach-Object {
        if (-not $_) { return }
        $row = $_ | ConvertFrom-Json
        if (-not $row.id) { return }
        if (($row.PSObject.Properties.Name -contains "baseline_wer") -or
            ($row.PSObject.Properties.Name -contains "baselines")) {
            $priorBaselines[[string]$row.id] = $row
        }
    }
}

$script = Join-Path (Split-Path $PSScriptRoot -Parent) "export\make_eval_set.py"
if (-not (Test-Path -LiteralPath $script)) {
    $script = Join-Path $PSScriptRoot "make_eval_set.py"
}
Write-Host "Building eval set (up to $Count utterances)..." -ForegroundColor Cyan
& $vpy $script --outdir $evalDir --count $Count
$ok = ($LASTEXITCODE -eq 0)
if ($ok -and (Test-Path -LiteralPath $generatedJsonl)) {
    Copy-Item -LiteralPath $generatedJsonl -Destination $trackedJsonl -Force
    Remove-Item -LiteralPath $generatedJsonl -Force -ErrorAction SilentlyContinue
    $evalPath = $trackedJsonl
}

if (-not $ok) {
    Write-Host "Dataset build failed; falling back to jfk.wav clip." -ForegroundColor Yellow
    $jfk = Join-Path $root "artifacts\workloads\speech\jfk.wav"
    if (-not (Test-Path $jfk)) { & (Join-Path $PSScriptRoot "get-audio.ps1") }
    $ref = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
    $rec = [ordered]@{ id = "jfk"; audio = "artifacts/workloads/speech/jfk.wav"; ref = $ref; duration_s = 11.0 }
    ($rec | ConvertTo-Json -Compress) | Set-Content -Path $trackedJsonl -Encoding UTF8
    $evalPath = $trackedJsonl
    Write-Host "Wrote 1 utterance (jfk) to src\workloads\eval\eval.jsonl" -ForegroundColor Green
}

if ($priorBaselines.Count -gt 0 -and (Test-Path -LiteralPath $evalPath)) {
    $merged = 0
    $lines = Get-Content -LiteralPath $evalPath -Encoding UTF8 | ForEach-Object {
        if (-not $_) { return $_ }
        $row = $_ | ConvertFrom-Json
        $prior = $priorBaselines[[string]$row.id]
        if ($prior) {
            foreach ($field in @("baseline_hyp", "baseline_wer", "baseline_cer", "baseline_lang",
                                 "baseline_task", "baseline_source", "baselines")) {
                if ($prior.PSObject.Properties.Name -contains $field) {
                    $row | Add-Member -NotePropertyName $field -NotePropertyValue $prior.$field -Force
                }
            }
            $merged++
        }
        ($row | ConvertTo-Json -Compress -Depth 8)
    }
    $lines | Set-Content -LiteralPath $evalPath -Encoding UTF8
    Write-Host "Preserved baked baselines for $merged clip(s)." -ForegroundColor DarkCyan
}

exit 0
