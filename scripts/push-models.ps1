# Uploads the local models/ tree to a (private) Hugging Face model repo.
# Models are gitignored and never committed to git; this snapshots them to HF,
# which gives 100GB free private storage and no LFS bandwidth throttling.
#
#   .\scripts\push-models.ps1
#   .\scripts\push-models.ps1 -Repo my-user/other-repo
#   .\scripts\push-models.ps1 -Message "re-export int4 variant"
#   .\scripts\push-models.ps1 -Public          # make/keep the repo public
#
# Requires a one-time login:  hf auth login   (token needs Write permission)

[CmdletBinding()]
param(
    [string]$Repo = "odobias/npu-hal-over-9000",
    [string]$Message = "Update models",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path $PSScriptRoot -Parent
$models = Join-Path $root "models"
if (-not (Test-Path $models)) { Write-Host "No models/ directory at $models" -ForegroundColor Red; exit 1 }

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

$visibility = if ($Public) { "--public" } else { "--private" }
$vLabel = if ($Public) { "public" } else { "private" }
Write-Host "Ensuring $vLabel repo $repoId exists..." -ForegroundColor Cyan
& $hf repos create $repoId --type model $visibility --exist-ok
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Uploading $models -> $repoId (this can take a while for ~1.5GB)..." -ForegroundColor Cyan
& $hf upload $repoId $models . --repo-type model --exclude ".cache/*" --commit-message $Message
if ($LASTEXITCODE -ne 0) { Write-Host "Upload failed." -ForegroundColor Red; exit 1 }

Write-Host "Models pushed: https://huggingface.co/$repoId" -ForegroundColor Green
