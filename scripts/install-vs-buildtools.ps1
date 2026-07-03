$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$installer = Join-Path $env:TEMP "vs_BuildTools.exe"
$url = "https://aka.ms/vs/17/release/vs_BuildTools.exe"

if (-not (Test-Path $installer)) {
    Write-Host "Downloading VS 2022 Build Tools bootstrapper..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
}

$args = @(
    "--wait",
    "--passive",
    "--norestart",
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--includeRecommended"
)

Write-Host "Launching installer (approve UAC if prompted)..." -ForegroundColor Yellow
$p = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
if ($p.ExitCode -ne 0) {
    Write-Host "Installer exit code: $($p.ExitCode)" -ForegroundColor Red
    exit $p.ExitCode
}

$vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$path = & $vsw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $path) {
    Write-Host "Build Tools finished but VC tools not detected." -ForegroundColor Red
    exit 1
}

Write-Host "VS 2022 Build Tools ready: $path" -ForegroundColor Green
