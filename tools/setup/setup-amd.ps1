# Ensures the AMD Ryzen AI SDK (VitisAI EP toolchain) and a compatible NPU driver
# are installed, so the AMD backend can build and run. Mirrors setup-intel.ps1:
# idempotent, reuse-or-install, and everything transient lands under third_party\.
#
#   .\setup-amd.ps1
#   .\setup-amd.ps1 -SkipDriver               # SDK only (driver already current)
#   .\setup-amd.ps1 -InstallDir "D:\RyzenAI\1.8.0-beta"
#
# NOTE: AMD ships no silent installer for the Ryzen AI SDK or the NPU driver, and
# both need elevation. This script downloads them, launches the vendor installers
# elevated (you approve the UAC + wizard prompts), and then polls for completion.
# Re-running after a successful install is a fast no-op.
[CmdletBinding()]
param(
    [string]$Version = "1.8.0-beta",
    [string]$InstallDir = "",
    [string]$CondaEnv = "",
    [string]$SdkUrl = "https://download.amd.com/opendownload/RyzenAI/1.8.0b0/ryzen-ai-lt-1.8.0-beta.exe",
    [string]$DriverUrl = "https://download.amd.com/opendownload/RyzenAI/1.8.0b0/NPU_RAI_376_WHQL.zip",
    [version]$MinDriverVersion = "32.0.20101.3760",
    [string]$DownloadDir = "",
    [switch]$SkipDriver,
    [int]$TimeoutMinutes = 45
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
# curl.exe on some boxes fails these AMD hosts with a TLS error (exit 35); the .NET
# stack negotiates fine once we opt into modern TLS.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $InstallDir)  { $InstallDir  = "C:\Program Files\RyzenAI\$Version" }
if (-not $CondaEnv)    { $CondaEnv    = "ryzen-ai-$Version" }
if (-not $DownloadDir) { $DownloadDir = Join-Path $root "third_party\amd-ryzenai\_download" }

# --- helpers ----------------------------------------------------------------

function Find-Conda {
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in @(
            "C:\ProgramData\miniforge3\Scripts\conda.exe",
            "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
            "C:\ProgramData\miniconda3\Scripts\conda.exe",
            "$env:USERPROFILE\miniconda3\Scripts\conda.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Ensure-Conda {
    $conda = Find-Conda
    if ($conda) { Write-Host "  conda: $conda" -ForegroundColor Green; return $conda }
    Write-Host "  conda not found; installing Miniforge3 via winget..." -ForegroundColor Yellow
    winget install --id CondaForge.Miniforge3 -e --accept-source-agreements --accept-package-agreements 2>$null
    $conda = Find-Conda
    if (-not $conda) {
        throw "conda is required by the Ryzen AI installer but could not be installed. Install Miniforge/Miniconda, then re-run."
    }
    return $conda
}

function Get-CondaEnvs {
    param([string]$Conda)
    (& $Conda env list 2>$null) | ForEach-Object { ($_ -split '\s+')[0] } | Where-Object { $_ -and $_ -notmatch '^#' }
}

function Get-NpuDriverVersion {
    $d = Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
        Where-Object { $_.DeviceName -match 'NPU Compute' } | Select-Object -First 1
    if ($d) { return $d.DriverVersion }
    return $null
}

function Test-SdkInstalled {
    param([string]$Conda)
    $haveDir = (Test-Path (Join-Path $InstallDir "onnxruntime")) -or (Test-Path (Join-Path $InstallDir "deployment"))
    $haveEnv = $Conda -and ((Get-CondaEnvs -Conda $Conda) -contains $CondaEnv)
    return ($haveDir -and $haveEnv)
}

function Write-SdkMetadata {
    $metadataDir = Join-Path $root "third_party\amd-ryzenai"
    New-Item -ItemType Directory -Force -Path $metadataDir | Out-Null
    ([ordered]@{
        schema_version = 1
        vendor = "AMD"
        package = "Ryzen AI SDK"
        version = $Version
        install_dir = $InstallDir
        conda_environment = $CondaEnv
        npu_driver_version = Get-NpuDriverVersion
        source = "setup-amd.ps1 installer parameters and installed driver metadata"
    } | ConvertTo-Json -Depth 5) |
        Set-Content -LiteralPath (Join-Path $metadataDir "VERSION.json") -Encoding UTF8
}

function Download-File {
    param([string]$Uri, [string]$OutFile, [long]$ExpectedBytes = 0)
    if (Test-Path $OutFile) {
        $len = (Get-Item $OutFile).Length
        if ($ExpectedBytes -eq 0 -or $len -eq $ExpectedBytes) {
            Write-Host "  cached: $OutFile ($([math]::Round($len/1MB,1)) MB)" -ForegroundColor DarkGray
            return
        }
        Write-Host "  re-downloading (size mismatch): $OutFile" -ForegroundColor Yellow
        Remove-Item $OutFile -Force
    }
    Write-Host "  downloading $Uri" -ForegroundColor Cyan
    # BITS handles multi-GB files far better than buffering in memory; fall back to IWR.
    try {
        Import-Module BitsTransfer -ErrorAction Stop
        Start-BitsTransfer -Source $Uri -Destination $OutFile -DisplayName (Split-Path $Uri -Leaf)
    } catch {
        Write-Host "  BITS unavailable ($($_.Exception.Message)); using Invoke-WebRequest" -ForegroundColor Yellow
        Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing -TimeoutSec 600
    }
    if ($ExpectedBytes -gt 0 -and (Get-Item $OutFile).Length -ne $ExpectedBytes) {
        throw "Download size mismatch for $OutFile (got $((Get-Item $OutFile).Length), expected $ExpectedBytes)."
    }
}

# --- 0) fast path -----------------------------------------------------------

$conda = Find-Conda
if (Test-SdkInstalled -Conda $conda) {
    Write-Host "Ryzen AI $Version already installed ($InstallDir, env '$CondaEnv')." -ForegroundColor Green
    if (-not [Environment]::GetEnvironmentVariable('RYZEN_AI_INSTALLATION_PATH', 'User')) {
        [Environment]::SetEnvironmentVariable('RYZEN_AI_INSTALLATION_PATH', $InstallDir, 'User')
        Write-Host "  set RYZEN_AI_INSTALLATION_PATH (User) = $InstallDir" -ForegroundColor DarkGray
    }
    Write-SdkMetadata
    return
}

New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

# --- 1) NPU driver ----------------------------------------------------------

if (-not $SkipDriver) {
    $cur = Get-NpuDriverVersion
    if ($cur -and ([version]$cur -ge $MinDriverVersion)) {
        Write-Host "NPU driver $cur >= ${MinDriverVersion}: OK" -ForegroundColor Green
    } else {
        Write-Host "NPU driver: have '$cur', need >= $MinDriverVersion. Installing $($DriverUrl.Split('/')[-1])..." -ForegroundColor Yellow
        $zip = Join-Path $DownloadDir (Split-Path $DriverUrl -Leaf)
        Download-File -Uri $DriverUrl -OutFile $zip
        $ex = Join-Path $DownloadDir "driver"
        Remove-Item $ex -Recurse -Force -ErrorAction SilentlyContinue
        Expand-Archive -Path $zip -DestinationPath $ex -Force
        $inst = Get-ChildItem $ex -Recurse -Filter "npu_sw_installer.exe" | Select-Object -First 1
        if (-not $inst) { throw "npu_sw_installer.exe not found in driver package." }
        Write-Host "  launching driver installer elevated -- approve the UAC prompt..." -ForegroundColor Cyan
        Start-Process -FilePath $inst.FullName -Verb RunAs
        Write-Host "  waiting for driver >= $MinDriverVersion (up to 10 min)..." -ForegroundColor Cyan
        $deadline = (Get-Date).AddMinutes(10)
        do {
            Start-Sleep -Seconds 15
            $cur = Get-NpuDriverVersion
        } while ((-not $cur -or [version]$cur -lt $MinDriverVersion) -and (Get-Date) -lt $deadline)
        if ($cur -and [version]$cur -ge $MinDriverVersion) {
            Write-Host "  NPU driver now $cur" -ForegroundColor Green
        } else {
            Write-Host "  driver still '$cur' -- finish the installer (a reboot may be required), then re-run." -ForegroundColor Yellow
        }
    }
}

# --- 2) SDK -----------------------------------------------------------------

$conda = Ensure-Conda

# Older conda has no 'tos' subcommand (and thus no TOS gate); newer conda will hang
# the installer's "Creating Conda env" step unless the channel TOS are accepted.
foreach ($ch in @("https://repo.anaconda.com/pkgs/main", "https://repo.anaconda.com/pkgs/r", "https://repo.anaconda.com/pkgs/msys2")) {
    & $conda tos accept --override-channels --channel $ch 2>$null | Out-Null
}

$exe = Join-Path $DownloadDir (Split-Path $SdkUrl -Leaf)
Download-File -Uri $SdkUrl -OutFile $exe
Write-Host "Launching Ryzen AI installer elevated -- approve UAC, then in the wizard:" -ForegroundColor Cyan
Write-Host "  * accept the license" -ForegroundColor Gray
Write-Host "  * keep the default install folder ($InstallDir)" -ForegroundColor Gray
Write-Host "  * keep the default conda env name ($CondaEnv)" -ForegroundColor Gray
Start-Process -FilePath $exe -Verb RunAs

Write-Host "Waiting for install to complete (env '$CondaEnv' + $InstallDir), up to $TimeoutMinutes min..." -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
do {
    Start-Sleep -Seconds 20
    $done = Test-SdkInstalled -Conda $conda
} while (-not $done -and (Get-Date) -lt $deadline)
if (-not $done) {
    throw "Ryzen AI install did not complete within $TimeoutMinutes min. Finish the wizard, then re-run setup-amd.ps1."
}

# --- 3) verify + wire up ----------------------------------------------------

Write-Host "Verifying VitisAI Execution Provider in '$CondaEnv'..." -ForegroundColor Cyan
$probe = & $conda run -n $CondaEnv python -c "import onnxruntime as ort; ps=ort.get_available_providers(); print('ORT', ort.__version__); print('VitisAI', 'VitisAIExecutionProvider' in ps)" 2>&1
$probe | ForEach-Object { Write-Host "  $_" }
if ($probe -notmatch 'VitisAI True') {
    Write-Host "  WARNING: VitisAIExecutionProvider not reported available in '$CondaEnv'." -ForegroundColor Yellow
}

if (-not [Environment]::GetEnvironmentVariable('RYZEN_AI_INSTALLATION_PATH', 'User')) {
    [Environment]::SetEnvironmentVariable('RYZEN_AI_INSTALLATION_PATH', $InstallDir, 'User')
    Write-Host "  set RYZEN_AI_INSTALLATION_PATH (User) = $InstallDir" -ForegroundColor DarkGray
}
Write-SdkMetadata

Write-Host ""
Write-Host "Ryzen AI $Version ready." -ForegroundColor Green
Write-Host "  install : $InstallDir" -ForegroundColor White
Write-Host "  condaenv: $CondaEnv" -ForegroundColor White
Write-Host "  build   : .\tools\build\build.ps1 -EnableAmd -DisableIntel -RyzenAiDir `"$InstallDir`"" -ForegroundColor White
