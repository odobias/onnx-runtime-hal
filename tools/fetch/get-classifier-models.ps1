# Downloads classifier models (FakeAudio / TSC) from the private Hugging Face
# mirror into artifacts/workloads/classifiers/. HF remote keys stay under deepfake/*;
# local runtime paths are remapped. Output is gitignored.
#
#   .\tools\fetch\get-classifier-models.ps1
#   .\tools\fetch\get-classifier-models.ps1 -Models tsc

[CmdletBinding()]
param(
    [ValidateSet("fakeaudio", "tsc", "all")]
    [string[]]$Models = @("all"),
    [string]$Repo = "odobias/npu-hal-over-9000"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$outRoot = Join-Path $root "artifacts\workloads\classifiers"
$staging = Join-Path $root "artifacts\build\classifier-asset-staging"

if ($Models -contains "all") { $Models = @("fakeaudio", "tsc") }

$hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
if (-not $hf) {
    $venvHf = Join-Path $root "artifacts/venv\Scripts\hf.exe"
    if (Test-Path $venvHf) { $hf = $venvHf }
}
if (-not $hf) {
    Write-Host "hf CLI not found. Install with: pip install -U 'huggingface_hub[cli]'" -ForegroundColor Red
    exit 1
}

$repoId = $Repo
if ($repoId -notmatch "/") {
    $ns = (& python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $ns) {
        Write-Host "Not logged in (or can't resolve user). Run: hf auth login   -- or pass -Repo <user>/<name>." -ForegroundColor Red
        exit 1
    }
    $repoId = "$($ns.Trim())/$Repo"
}

$includes = @()
if ($Models -contains "fakeaudio") {
    # Whole FakeAudio graph + NPU split variants (frontend / backbone / expanded bias).
    $includes += "deepfake/fakeaudio/**"
    # Nested test_audio/ (Šimon export smoke) plus legacy flat YouTube WAVs.
    $includes += "deepfake/audio-samples/**"
    $includes += "deepfake/fixtures/fakeaudio/**"
    $includes += "deepfake/fixtures/fakeaudio-bb-fp32/**"
    $includes += "deepfake/fixtures/fakeaudio-bb-expanded-attention-bias/**"
}
if ($Models -contains "tsc") {
    $includes += "deepfake/tsc/*"
    $includes += "deepfake/fixtures/tsc/*"
}

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force -Path $staging | Out-Null

Write-Host "Downloading $($Models -join ', ') from $repoId -> staging ..." -ForegroundColor Cyan
$dlArgs = @($repoId, "--repo-type", "model", "--local-dir", $staging)
foreach ($p in $includes) { $dlArgs += "--include=$p" }
& $hf download @dlArgs
if ($LASTEXITCODE -ne 0) { Write-Host "Download failed." -ForegroundColor Red; exit 1 }

function Move-Mapped([string]$HfRel, [string]$LocalRel) {
    $src = Join-Path $staging ($HfRel -replace '/', '\')
    $dst = Join-Path $root ($LocalRel -replace '/', '\')
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Host "  ! not in repo, skipping: $HfRel" -ForegroundColor DarkYellow
        return
    }
    $dstParent = Split-Path $dst -Parent
    New-Item -ItemType Directory -Force -Path $dstParent | Out-Null
    if (Test-Path -LiteralPath $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
    Move-Item -LiteralPath $src -Destination $dst -Force
    Write-Host "  $HfRel -> $LocalRel" -ForegroundColor DarkGray
}

if ($Models -contains "fakeaudio") {
    # Preserve Šimon's test_audio kit across HF mirror refreshes (legacy YouTube
    # WAVs still live at the audio-samples root from the mirror).
    $testAudioKeep = Join-Path $outRoot "audio-samples\test_audio"
    $testAudioTmp = Join-Path $root "artifacts\scratch\test_audio_preserve"
    $hadTestAudio = Test-Path -LiteralPath $testAudioKeep
    if ($hadTestAudio) {
        if (Test-Path $testAudioTmp) { Remove-Item $testAudioTmp -Recurse -Force }
        New-Item -ItemType Directory -Force -Path (Split-Path $testAudioTmp) | Out-Null
        Move-Item -LiteralPath $testAudioKeep -Destination $testAudioTmp -Force
    }
    Move-Mapped "deepfake/fakeaudio" "artifacts/workloads/classifiers/fakeaudio"
    Move-Mapped "deepfake/audio-samples" "artifacts/workloads/classifiers/audio-samples"
    Move-Mapped "deepfake/fixtures/fakeaudio" "artifacts/workloads/classifiers/fixtures/fakeaudio"
    Move-Mapped "deepfake/fixtures/fakeaudio-bb-fp32" "artifacts/workloads/classifiers/fixtures/fakeaudio-bb-fp32"
    Move-Mapped "deepfake/fixtures/fakeaudio-bb-expanded-attention-bias" "artifacts/workloads/classifiers/fixtures/fakeaudio-bb-expanded-attention-bias"
    if ($hadTestAudio) {
        $restore = Join-Path $outRoot "audio-samples\test_audio"
        if (Test-Path $restore) { Remove-Item $restore -Recurse -Force }
        Move-Item -LiteralPath $testAudioTmp -Destination $restore -Force
        Write-Host "  restored audio-samples/test_audio (wanna-deepfk export smoke)" -ForegroundColor DarkCyan
    } else {
        Write-Host "  tip: .\tools\fetch\get-fakeaudio-test-audio.ps1  # balanced 5+5 smoke set" -ForegroundColor DarkGray
    }
}
if ($Models -contains "tsc") {
    Move-Mapped "deepfake/tsc" "artifacts/workloads/classifiers/tsc"
    Move-Mapped "deepfake/fixtures/tsc" "artifacts/workloads/classifiers/fixtures/tsc"
}

# Prefer locally exported NPU-split artifacts when already present under legacy cache.
$legacyFa = Join-Path $root "models\deepfake\fakeaudio"
$dstFa = Join-Path $outRoot "fakeaudio"
if (Test-Path -LiteralPath $legacyFa) {
    foreach ($name in @(
        "model.backbone-fp32.onnx",
        "model.frontend-fp32.onnx",
        "model.backbone.expanded-attention-bias.onnx"
    )) {
        $from = Join-Path $legacyFa $name
        $to = Join-Path $dstFa $name
        if ((Test-Path -LiteralPath $from) -and -not (Test-Path -LiteralPath $to)) {
            New-Item -ItemType Directory -Force -Path $dstFa | Out-Null
            Copy-Item -LiteralPath $from -Destination $to -Force
            Write-Host "  legacy cache -> artifacts/workloads/classifiers/fakeaudio/$name" -ForegroundColor DarkGray
        }
    }
}
$legacyFix = Join-Path $root "models\deepfake\fixtures"
$dstFix = Join-Path $outRoot "fixtures"
if (Test-Path -LiteralPath $legacyFix) {
    foreach ($name in @("fakeaudio-bb-fp32", "fakeaudio-bb-expanded-attention-bias")) {
        $from = Join-Path $legacyFix $name
        $to = Join-Path $dstFix $name
        if ((Test-Path -LiteralPath $from) -and -not (Test-Path -LiteralPath $to)) {
            Copy-Item -LiteralPath $from -Destination $to -Recurse -Force
            Write-Host "  legacy cache -> artifacts/workloads/classifiers/fixtures/$name" -ForegroundColor DarkGray
        }
    }
}

Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue

function Get-Size([string]$Dir) {
    if (-not (Test-Path $Dir)) { return 0 }
    [math]::Round(((Get-ChildItem $Dir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
}

if ($Models -contains "fakeaudio") {
    Write-Host "FakeAudio ready: $(Join-Path $outRoot 'fakeaudio') ($(Get-Size (Join-Path $outRoot 'fakeaudio')) MB)" -ForegroundColor Green
}
if ($Models -contains "tsc") {
    Write-Host "TSC ready: $(Join-Path $outRoot 'tsc') ($(Get-Size (Join-Path $outRoot 'tsc')) MB)" -ForegroundColor Green
}
Write-Host "Total artifacts/workloads/classifiers: $(Get-Size $outRoot) MB" -ForegroundColor Green
