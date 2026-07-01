# Downloads a public-domain 16 kHz mono sample (JFK, from whisper.cpp) into models/.
# Kept out of git; fetched on demand so the repo stays light.
[CmdletBinding()]
param(
    [string]$Url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null
$dest = Join-Path $models "jfk.wav"

if (Test-Path $dest) {
    Write-Host "Audio already present: $dest" -ForegroundColor Green
    return
}

Write-Host "Downloading sample audio -> $dest" -ForegroundColor Cyan
Invoke-WebRequest -Uri $Url -OutFile $dest -UseBasicParsing
Write-Host "Done ($([math]::Round((Get-Item $dest).Length/1kb)) KB)." -ForegroundColor Green
