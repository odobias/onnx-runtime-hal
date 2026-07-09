# Downloads the proprietary deepfake-detection pipeline models (FakeAudio /
# Generated Audio Detector, Text Scam Classifier, and the sherpa-exported
# Whisper tiny.en used by that pipeline) from the internal Avast Artifactory
# into models/deepfake/. Output is gitignored (models/ is never committed) --
# a fresh clone/machine re-runs this script instead of the binaries living in
# git. These are internal proprietary artifacts: do NOT push them anywhere
# public (they are intentionally excluded from scripts/push-models.ps1's
# scope -- that script snapshots ASR benchmark variants only).
#
# Auth: Windows Integrated Auth against artifactory.ida.avast.com. Must be run
# on a machine joined to / VPN'd into the corporate network with a domain
# account that has read access to the ai-models-generic-local repo.
#
#   .\scripts\get-deepfake-models.ps1                 # fetch everything
#   .\scripts\get-deepfake-models.ps1 -Models fakeaudio,tsc
#   .\scripts\get-deepfake-models.ps1 -Models whisper -SkipInt8

[CmdletBinding()]
param(
    [ValidateSet("fakeaudio", "tsc", "whisper", "all")]
    [string[]]$Models = @("all"),
    [switch]$SkipInt8
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
$outRoot = Join-Path (Join-Path $root "models") "deepfake"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

if ($Models -contains "all") { $Models = @("fakeaudio", "tsc", "whisper") }

$base = "https://artifactory.ida.avast.com/artifactory/ai-models-generic-local/vertex-ai/ppp-ctores-deepfk-ai-f6/europe-west1"

function Get-File {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$OutFile
    )
    if (Test-Path $OutFile) {
        Write-Host "Already present: $OutFile" -ForegroundColor DarkGray
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $OutFile -Parent) | Out-Null
    Write-Host "Downloading $Uri" -ForegroundColor Cyan
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseDefaultCredentials
}

function Get-Size {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) { return 0 }
    [math]::Round(((Get-ChildItem $Dir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
}

if ($Models -contains "fakeaudio") {
    # "Generated Audio Detector" (GAD): MS-CLAP audio embedder + classifier head,
    # trained/exported 2025-05. CPU/GPU-validated in production; NPU untested.
    $dir = Join-Path $outRoot "fakeaudio"
    Get-File "$base/audio-detector-model-onnx/1/model.onnx" (Join-Path $dir "model.onnx")
    Write-Host "FakeAudio (GAD) ready: $dir ($(Get-Size $dir) MB)" -ForegroundColor Green
}

if ($Models -contains "tsc") {
    # Text Scam Classifier: DistilBERT-based, ONNX export v2 (latest). Targets
    # NPU on ARM (QNN) and Intel NPU via C++.
    $dir = Join-Path $outRoot "tsc"
    Get-File "$base/text-scam-classifier-model-onnx/2/model.onnx" (Join-Path $dir "model.onnx")
    Get-File "$base/text-scam-classifier-model-onnx/2/vocab.json" (Join-Path $dir "vocab.json")
    Get-File "$base/text-scam-classifier-model-onnx/2/merges.txt" (Join-Path $dir "merges.txt")
    Write-Host "TSC ready: $dir ($(Get-Size $dir) MB)" -ForegroundColor Green
}

if ($Models -contains "whisper") {
    # sherpa-onnx export of whisper-tiny.en used by the deepfake pipeline (distinct
    # from this repo's own OpenVINO GenAI IR export under models/variants/) --
    # the overview calls this the strongest NPU candidate of the three models.
    $dir = Join-Path $outRoot "whisper-tiny-en-sherpa"
    Get-File "$base/whisper-tiny-en-model-onnx/2/tiny.en-encoder.onnx" (Join-Path $dir "tiny.en-encoder.onnx")
    Get-File "$base/whisper-tiny-en-model-onnx/2/tiny.en-decoder.onnx" (Join-Path $dir "tiny.en-decoder.onnx")
    Get-File "$base/whisper-tiny-en-model-onnx/2/tiny.en-tokens.txt" (Join-Path $dir "tiny.en-tokens.txt")
    Write-Host "Whisper tiny.en (sherpa, fp32) ready: $dir ($(Get-Size $dir) MB)" -ForegroundColor Green

    if (-not $SkipInt8) {
        $dirInt8 = Join-Path $outRoot "whisper-tiny-en-sherpa-int8"
        Get-File "$base/whisper-tiny-en-int8-model-onnx/2/tiny.en-encoder.int8.onnx" (Join-Path $dirInt8 "tiny.en-encoder.int8.onnx")
        Get-File "$base/whisper-tiny-en-int8-model-onnx/2/tiny.en-decoder.int8.onnx" (Join-Path $dirInt8 "tiny.en-decoder.int8.onnx")
        Get-File "$base/whisper-tiny-en-int8-model-onnx/2/tiny.en-tokens.txt" (Join-Path $dirInt8 "tiny.en-tokens.txt")
        Write-Host "Whisper tiny.en (sherpa, int8) ready: $dirInt8 ($(Get-Size $dirInt8) MB)" -ForegroundColor Green
    }
}

Write-Host "Total models/deepfake: $(Get-Size $outRoot) MB" -ForegroundColor Green
