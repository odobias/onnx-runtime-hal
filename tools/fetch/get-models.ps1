# Downloads benchmark assets from the private Hugging Face repository into the
# workload-first paths declared by asset-map.json.
# Mirror of push-models.ps1. Run this after a fresh clone to pull the weights,
# since models/ is gitignored and never committed.
#
# The HF repo and the local models/ tree share one clean, browsable layout
# (whisper/en-static-onnx, whisper/amd, whisper/variants-ov/fp32, audio/jfk.wav,
# eval/, deepfake/, ...). The canonical map lives in scripts/asset-map.json;
# this script downloads each hf path to its local target (identity today, but the
# indirection means a future layout change is a one-file edit in the map).
#
#   .\tools\fetch\get-models.ps1
#   .\tools\fetch\get-models.ps1 -Repo my-user/other-repo
#
# For a private repo you must first:  hf auth login

[CmdletBinding()]
param(
    [string]$Repo = "odobias/npu-hal-over-9000"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

$hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
if (-not $hf) {
    $venvHf = Join-Path $root "artifacts/venv\Scripts\hf.exe"
    if (Test-Path $venvHf) { $hf = $venvHf }
}
if (-not $hf) { Write-Host "hf CLI not found. Install with: pip install -U 'huggingface_hub[cli]'" -ForegroundColor Red; exit 1 }

# Qualify a bare repo name with the logged-in username.
$repoId = $Repo
if ($repoId -notmatch "/") {
    $ns = (& python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $ns) {
        Write-Host "Not logged in (or can't resolve user). Run: hf auth login   -- or pass -Repo <user>/<name>." -ForegroundColor Red
        exit 1
    }
    $repoId = "$($ns.Trim())/$Repo"
}

$mapPath = Join-Path $PSScriptRoot "asset-map.json"
if (-not (Test-Path $mapPath)) { Write-Host "Missing location map: $mapPath" -ForegroundColor Red; exit 1 }
$entries = (Get-Content $mapPath -Raw | ConvertFrom-Json).map

# Pull every mapped HF path in one shot into a staging tree, then remap into the
# stable local layout. Staging keeps hf's repo-relative structure so we can move
# each hf-path to its local target deterministically.
$staging = Join-Path $root "artifacts\build\asset-staging"
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force -Path $staging | Out-Null

$includes = foreach ($e in $entries) {
    if ($e.kind -eq "file") { $e.hf } else { "$($e.hf)/**" }
}

Write-Host "Downloading $repoId (mapped subtree) -> staging ..." -ForegroundColor Cyan
$dlArgs = @($repoId, "--repo-type", "model", "--local-dir", $staging)
# Keep each filter in one native argument. Passing `--include`, `$p` separately lets
# Windows PowerShell expand `**` against this checkout before `hf` sees it (for
# example src/workloads/classifiers/** became ort_classifier.cpp).
foreach ($p in $includes) { $dlArgs += "--include=$p" }
& $hf download @dlArgs
if ($LASTEXITCODE -ne 0) { Write-Host "Download failed." -ForegroundColor Red; exit 1 }

foreach ($e in $entries) {
    $src = Join-Path $staging ($e.hf -replace '/', '\')
    $dst = Join-Path $root ($e.local -replace '/', '\')
    if (-not (Test-Path $src)) {
        Write-Host "  ! not in repo, skipping: $($e.hf)" -ForegroundColor DarkYellow
        continue
    }
    $dstParent = Split-Path $dst -Parent
    New-Item -ItemType Directory -Force -Path $dstParent | Out-Null
    if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
    Move-Item -LiteralPath $src -Destination $dst -Force
    Write-Host "  $($e.hf)  ->  $($e.local -replace '/', '\')" -ForegroundColor DarkGray
}

Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Benchmark assets ready at their mapped workload/model paths." -ForegroundColor Green
