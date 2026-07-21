# Stage AvastClient-shaped product models into artifacts/workloads for the
# portable suite (onnx-sherpa Whisper + classifier fixtures).
#
# Sources (first hit wins):
#   1) models/deepfake/* (local cache / HF pull)
#   2) optional -MlRoot (e.g. MLM install with \whisper, \fakeaudio, \tsc)
#
#   .\tools\fetch\stage-product-models.ps1
#   .\tools\fetch\stage-product-models.ps1 -MlRoot C:\Sources\models

[CmdletBinding()]
param(
    [string]$MlRoot = "",
    [switch]$SkipClassifiers,
    [switch]$SkipWhisper
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

function Copy-FileForced {
    param([string]$Src, [string]$Dst)
    if (-not (Test-Path -LiteralPath $Src)) { throw "Missing source: $Src" }
    $dir = Split-Path $Dst -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
    Write-Host "  $Src -> $Dst" -ForegroundColor DarkGray
}

function Copy-TreeForced {
    param([string]$Src, [string]$Dst)
    if (-not (Test-Path -LiteralPath $Src)) { throw "Missing source: $Src" }
    New-Item -ItemType Directory -Force -Path $Dst | Out-Null
    Copy-Item -LiteralPath (Join-Path $Src "*") -Destination $Dst -Recurse -Force
    Write-Host "  $Src\ -> $Dst\" -ForegroundColor DarkGray
}

if (-not $SkipWhisper) {
    Write-Host "== Staging Sherpa-export Whisper (product names) ==" -ForegroundColor Cyan
    $dst = Join-Path $root "artifacts\workloads\whisper\models\sherpa-export"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null

    $candidates = @(
        @{
            enc = Join-Path $root "models\deepfake\whisper-tiny-en-sherpa\tiny.en-encoder.onnx"
            dec = Join-Path $root "models\deepfake\whisper-tiny-en-sherpa\tiny.en-decoder.onnx"
            tok = Join-Path $root "models\deepfake\whisper-tiny-en-sherpa\tiny.en-tokens.txt"
        }
    )
    if ($MlRoot) {
        $candidates = @(
            @{
                enc = Join-Path $MlRoot "whisper\encoder.onnx"
                dec = Join-Path $MlRoot "whisper\decoder.onnx"
                tok = Join-Path $MlRoot "whisper\tokens.txt"
            }
        ) + $candidates
    }

    $picked = $null
    foreach ($c in $candidates) {
        if ((Test-Path -LiteralPath $c.enc) -and (Test-Path -LiteralPath $c.dec) -and
            (Test-Path -LiteralPath $c.tok)) {
            $picked = $c
            break
        }
    }
    if (-not $picked) {
        throw "No Sherpa Whisper model found under models/deepfake/whisper-tiny-en-sherpa or -MlRoot\whisper"
    }

    Copy-FileForced $picked.enc (Join-Path $dst "encoder.onnx")
    Copy-FileForced $picked.dec (Join-Path $dst "decoder.onnx")
    Copy-FileForced $picked.tok (Join-Path $dst "tokens.txt")
    Write-Host "Whisper staged: $dst" -ForegroundColor Green
}

if (-not $SkipClassifiers) {
    Write-Host "== Staging classifier models + fixtures ==" -ForegroundColor Cyan
    $migrate = Join-Path $root "tools\fetch\migrate-classifier-layout.ps1"
    if (Test-Path -LiteralPath $migrate) {
        & $migrate
    } else {
        throw "Missing $migrate"
    }

    if ($MlRoot) {
        $faSrc = Join-Path $MlRoot "fakeaudio\model.onnx"
        $tscSrc = Join-Path $MlRoot "tsc\model.onnx"
        if (Test-Path -LiteralPath $faSrc) {
            Copy-FileForced $faSrc (Join-Path $root "artifacts\workloads\classifiers\fakeaudio\model.onnx")
        }
        if (Test-Path -LiteralPath $tscSrc) {
            Copy-FileForced $tscSrc (Join-Path $root "artifacts\workloads\classifiers\tsc\model.onnx")
            foreach ($extra in @("vocab.json", "merges.txt")) {
                $src = Join-Path $MlRoot "tsc\$extra"
                if (Test-Path -LiteralPath $src) {
                    Copy-FileForced $src (Join-Path $root "artifacts\workloads\classifiers\tsc\$extra")
                }
            }
        }
    }
    Write-Host "Classifiers staged under artifacts\workloads\classifiers" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
