[CmdletBinding()]
param(
    [string]$Python = "C:\ProgramData\miniforge3\envs\ryzen-ai-1.8.0-beta\python.exe",
    [switch]$ReuseCache
)

$ErrorActionPreference = "Stop"
chcp 65001 *> $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if ($null -ne $PSStyle -and $null -ne $PSStyle.OutputRendering) {
    $PSStyle.OutputRendering = "Ansi"
}

if (-not (Test-Path $Python)) {
    throw "Ryzen AI Python not found: $Python"
}

$script = Join-Path $PSScriptRoot "repro.py"
Write-Host "=== CPU control (expected: exit 0, output [64,64,96]) ==="
& $Python $script cpu
if ($LASTEXITCODE -ne 0) {
    throw "CPU control failed with exit code $LASTEXITCODE"
}

Write-Host "`n=== VitisAI (expected on affected runtime: session succeeds, first inference crashes with 0xC0000005) ==="
$arguments = @($script, "vitisai")
if ($ReuseCache) {
    $arguments += "--reuse-cache"
}
& $Python @arguments
$code = $LASTEXITCODE
$unsigned = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$code), 0)
Write-Host ("VitisAI process exit code: {0} (0x{1:X8})" -f $code, $unsigned)
exit $code
