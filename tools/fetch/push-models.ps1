# Uploads workload assets to a private Hugging Face model repository.
# Model payloads are gitignored and never committed to git; this snapshots them to HF,
# which gives 100GB free private storage and no LFS bandwidth throttling.
#
# The HF repo and local models/ share one clean layout; the canonical map is
# scripts/asset-map.json. This uploads each local path to its hf target
# (identity today), so get-models.ps1 and push-models.ps1 stay in lockstep.
#
#   .\tools\fetch\push-models.ps1
#   .\tools\fetch\push-models.ps1 -Repo my-user/other-repo
#   .\tools\fetch\push-models.ps1 -Message "re-export int4 variant"
#   .\tools\fetch\push-models.ps1 -Public          # make/keep the repo public
#
# Requires a one-time login:  hf auth login   (token needs Write permission)

[CmdletBinding()]
param(
    [string]$Repo = "gendigital/npu-hal-over-9000",
    [string]$Message = "Update models",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

$hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
$hfArgsPrefix = @()
if (-not $hf) {
    foreach ($candidate in @(
        (Join-Path $root "artifacts\venv\Scripts\hf.exe"),
        (Join-Path $root ".venv\Scripts\hf.exe")
    )) {
        if (Test-Path $candidate) { $hf = $candidate; break }
    }
}
if (-not $hf) {
    # Older venvs ship huggingface-cli only; invoke via python -m.
    $py = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
    if ($py) {
        $hf = $py
        $hfArgsPrefix = @("-m", "huggingface_hub.commands.huggingface_cli")
    }
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

$visibility = if ($Public) { "--public" } else { "--private" }
$vLabel = if ($Public) { "public" } else { "private" }
Write-Host "Ensuring $vLabel repo $repoId exists..." -ForegroundColor Cyan
# Newer `hf` CLI uses `repos create`; older huggingface-cli uses `repo create`.
$createOk = $false
foreach ($createCmd in @(
    (@("repos", "create", $repoId, "--type", "model", $visibility, "--exist-ok")),
    (@("repo", "create", $repoId, "--type", "model", $visibility, "--exist-ok"))
)) {
    & $hf @hfArgsPrefix @createCmd 2>$null
    if ($LASTEXITCODE -eq 0) { $createOk = $true; break }
}
if (-not $createOk) {
    # Repo may already exist; verify with a no-op listing via upload path later.
    Write-Host "  (repo create skipped/failed; continuing if $repoId already exists)" -ForegroundColor DarkYellow
}

# Upload each mapped local path to its HF target. Per-entry (rather than one blanket
# upload) is what lets the HF layout differ from local when the map isn't identity.
foreach ($e in $entries) {
    $localPath = Join-Path $root ($e.local -replace '/', '\')
    if (-not (Test-Path $localPath)) {
        Write-Host "  ! local missing, skipping: $($e.local -replace '/', '\')" -ForegroundColor DarkYellow
        continue
    }
    Write-Host "  $($e.local -replace '/', '\')  ->  $($e.hf)" -ForegroundColor DarkGray
    & $hf @hfArgsPrefix upload $repoId $localPath $e.hf --repo-type model --commit-message $Message
    if ($LASTEXITCODE -ne 0) { Write-Host "Upload failed for $($e.local)." -ForegroundColor Red; exit 1 }
}

Write-Host "Models pushed: https://huggingface.co/$repoId" -ForegroundColor Green
