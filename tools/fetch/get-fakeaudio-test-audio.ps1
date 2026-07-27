# Fetch Šimon Mandlík's FakeAudio export-smoke WAVs (5 real + 5 synthetic)
# from ai-research/wanna-deepfk-ai into artifacts/workloads/classifiers/audio-samples/test_audio/.
#
# These are the clips op_export.py logs against after ONNX export. Prefer them over the
# legacy YouTube "efficacy review" five for labeled smoke / fixture baking.
#
# Usage:
#   .\tools\fetch\get-fakeaudio-test-audio.ps1
#   .\tools\fetch\get-fakeaudio-test-audio.ps1 -SourceDir C:\path\to\test_audio
#
# Stage B (GCS msclap_2023 chunks) needs bucket ACL; not implemented here.

[CmdletBinding()]
param(
    [string]$SourceDir = "",
    [string]$RepoUrl = "https://git.int.avast.com/ai-research/wanna-deepfk-ai.git",
    [string]$RepoRef = "main",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$dest = Join-Path $root "artifacts\workloads\classifiers\audio-samples\test_audio"
$scratchRepo = Join-Path $root "artifacts\scratch\wanna-deepfk-ai"
$relTestAudio = "src\wanna_deepfk_ai\audio\test_audio"

function Test-TestAudioLayout([string]$dir) {
    return (Test-Path (Join-Path $dir "real")) -and (Test-Path (Join-Path $dir "synthetic"))
}

function Copy-TestAudio([string]$src, [string]$dst) {
    if ((Test-Path $dst) -and -not $Force) {
        $have = @(Get-ChildItem -Recurse $dst -Filter *.wav -ErrorAction SilentlyContinue).Count
        if ($have -ge 10) {
            Write-Host "test_audio already present ($have wavs) at $dst (pass -Force to overwrite)" -ForegroundColor DarkCyan
            return
        }
    }
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item -Recurse (Join-Path $src "real") (Join-Path $dst "real")
    Copy-Item -Recurse (Join-Path $src "synthetic") (Join-Path $dst "synthetic")
    $n = @(Get-ChildItem -Recurse $dst -Filter *.wav).Count
    Write-Host "Copied $n wavs -> $dst" -ForegroundColor Green
}

if ($SourceDir) {
    if (-not (Test-TestAudioLayout $SourceDir)) {
        throw "SourceDir must contain real/ and synthetic/: $SourceDir"
    }
    Copy-TestAudio $SourceDir $dest
    exit 0
}

$cloned = Join-Path $scratchRepo $relTestAudio
if (-not (Test-TestAudioLayout $cloned)) {
    Write-Host "Cloning $RepoUrl (sparse test_audio)..." -ForegroundColor Cyan
    if (Test-Path $scratchRepo) { Remove-Item -Recurse -Force $scratchRepo }
    New-Item -ItemType Directory -Force -Path (Split-Path $scratchRepo) | Out-Null
    git clone --depth 1 --filter=blob:none --sparse $RepoUrl $scratchRepo
    if ($LASTEXITCODE -ne 0) { throw "git clone failed ($LASTEXITCODE)" }
    Push-Location $scratchRepo
    try {
        git sparse-checkout set "src/wanna_deepfk_ai/audio/test_audio"
        if ($LASTEXITCODE -ne 0) { throw "sparse-checkout failed ($LASTEXITCODE)" }
        if ($RepoRef -and $RepoRef -ne "main") {
            git fetch --depth 1 origin $RepoRef
            git checkout $RepoRef
        }
        # Materialize WAV blobs (filter=blob:none leaves stubs until checkout).
        git checkout HEAD -- "src/wanna_deepfk_ai/audio/test_audio"
    } finally {
        Pop-Location
    }
}

if (-not (Test-TestAudioLayout $cloned)) {
    throw "test_audio layout missing after clone: $cloned"
}

$probe = Join-Path $cloned "real\1.wav"
if ((Get-Item $probe).Length -lt 1000) {
    throw "WAV looks like an empty sparse stub: $probe — re-run without blob filter"
}

Copy-TestAudio $cloned $dest
Write-Host "Done. Next: .venv\Scripts\python.exe tools\validate\validate_fakeaudio_onnx.py" -ForegroundColor Green
Write-Host "         .venv\Scripts\python.exe tools\fixtures\generate.py --models fakeaudio" -ForegroundColor Green
