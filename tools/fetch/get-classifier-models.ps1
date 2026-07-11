# Downloads the deepfake-detection pipeline classifier models (FakeAudio /
# Generated Audio Detector, and the Text Scam Classifier) from the private
# Hugging Face model repo into workloads/classifiers/. Output is gitignored (models/
# is never committed) -- a fresh clone/machine re-runs this script instead of
# the binaries living in git.
#
# The models are mirrored to HF by scripts/push-models.ps1 (which snapshots the
# whole models/ tree). This script pulls only the deepfake/* subtree, so it does
# NOT touch any internal/corporate artifact repository.
#
# For a private repo you must first:  hf auth login
#
#   .\tools\fetch\get-classifier-models.ps1                    # fetch fakeaudio + tsc
#   .\tools\fetch\get-classifier-models.ps1 -Models tsc
#   .\tools\fetch\get-classifier-models.ps1 -Repo my-user/other-repo

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
# HF stores these under deepfake/... so downloading into models/ lands them at
# the paths the validators/benchmark expect (workloads/classifiers/...).
$outDir = Join-Path $root "models"
$outRoot = Join-Path $outDir "deepfake"

if ($Models -contains "all") { $Models = @("fakeaudio", "tsc") }

$hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
if (-not $hf) {
    # Fall back to the project venv's CLI so a fresh shell works without activating it.
    $venvHf = Join-Path $root ".venv\Scripts\hf.exe"
    if (Test-Path $venvHf) { $hf = $venvHf }
}
if (-not $hf) {
    Write-Host "hf CLI not found. Install with: pip install -U 'huggingface_hub[cli]'" -ForegroundColor Red
    exit 1
}

# Qualify a bare repo name with the logged-in username (mirrors get-models.ps1).
$repoId = $Repo
if ($repoId -notmatch "/") {
    $ns = (& python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $ns) {
        Write-Host "Not logged in (or can't resolve user). Run: hf auth login   -- or pass -Repo <user>/<name>." -ForegroundColor Red
        exit 1
    }
    $repoId = "$($ns.Trim())/$Repo"
}

# Per-model include globs (HF filtering is fnmatch-style; '*' spans '/').
$includes = @()
if ($Models -contains "fakeaudio") {
    # GAD model + the labeled real/deepfake clips its validator scores against.
    $includes += "deepfake/fakeaudio/*"
    $includes += "deepfake/audio-samples/*"
    # Pre-baked C++ benchmark fixtures (model-ready tensors + CPU reference p), if
    # present on HF, so benchmark-onnx.ps1 can run this classifier WITHOUT the Python
    # .venv. They are a snapshot of the validated preprocessing -- stale if the model
    # or preprocessing changes; regenerate with benchmark-onnx.ps1 -RegenerateFixtures.
    $includes += "deepfake/fixtures/fakeaudio/*"
}
if ($Models -contains "tsc") {
    # DistilBERT/RoBERTa-tokenized scam classifier: model + vocab/merges +
    # the validation_samples/{scam,clean}.txt the validator scores against.
    $includes += "deepfake/tsc/*"
    $includes += "deepfake/fixtures/tsc/*"   # pre-baked C++ fixtures (see note above)
}

Write-Host "Downloading $($Models -join ', ') from $repoId -> $outRoot ..." -ForegroundColor Cyan
$dlArgs = @($repoId, "--repo-type", "model", "--local-dir", $outDir)
foreach ($p in $includes) { $dlArgs += @("--include", $p) }
& $hf download @dlArgs
if ($LASTEXITCODE -ne 0) { Write-Host "Download failed." -ForegroundColor Red; exit 1 }

function Get-Size {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) { return 0 }
    [math]::Round(((Get-ChildItem $Dir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
}

if ($Models -contains "fakeaudio") {
    Write-Host "FakeAudio (GAD) ready: $(Join-Path $outRoot 'fakeaudio') ($(Get-Size (Join-Path $outRoot 'fakeaudio')) MB)" -ForegroundColor Green
}
if ($Models -contains "tsc") {
    Write-Host "TSC ready: $(Join-Path $outRoot 'tsc') ($(Get-Size (Join-Path $outRoot 'tsc')) MB)" -ForegroundColor Green
}
Write-Host "Total workloads/classifiers: $(Get-Size $outRoot) MB" -ForegroundColor Green
