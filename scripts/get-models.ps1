# Downloads the model snapshot from the (private) Hugging Face repo into models/.
# Mirror of push-models.ps1. Run this after a fresh clone to pull the weights,
# since models/ is gitignored and never committed.
#
#   .\scripts\get-models.ps1
#   .\scripts\get-models.ps1 -Repo my-user/other-repo
#
# For a private repo you must first:  hf auth login

[CmdletBinding()]
param(
    [string]$Repo = "odobias/npu-hal-over-9000",
    [string]$Out = "models"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root $Out

$hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
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

Write-Host "Downloading $repoId -> $outDir ..." -ForegroundColor Cyan
& $hf download $repoId --repo-type model --local-dir $outDir
if ($LASTEXITCODE -ne 0) { Write-Host "Download failed." -ForegroundColor Red; exit 1 }

Write-Host "Models ready: $outDir" -ForegroundColor Green
