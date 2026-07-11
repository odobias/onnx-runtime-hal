# Verifies the AMD build output in build\x64\Release is a self-contained package:
# runs the packaged exe with a PATH scrubbed of the Ryzen AI SDK (and everything
# else except the package dir + Windows System32), using the BUNDLED models next to
# the exe. If VitisAI/DirectML/CPU still work, the folder carries everything it needs.
#
#   .\test-selfcontained-amd.ps1            # full CPU/GPU/NPU sweep (NPU cold compiles)
#   .\test-selfcontained-amd.ps1 -HotOnly   # NPU-only, reuse populated cache (fast, quiet)
[CmdletBinding()]
param([switch]$HotOnly)
$ErrorActionPreference = "Continue"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pkg = Join-Path $root "build\x64\Release"
$exe = Join-Path $pkg "NpuInferenceBench.exe"
$sys = Join-Path $env:SystemRoot "System32"

# Minimal, SDK-free PATH: package dir first, then core Windows only.
$cleanPath = "$pkg;$sys;$env:SystemRoot;$sys\WindowsPowerShell\v1.0"

Write-Host "Package : $pkg" -ForegroundColor Cyan
Write-Host "Scrubbed PATH = $cleanPath" -ForegroundColor DarkGray
Write-Host ("RYZEN_AI_INSTALLATION_PATH (proc) = {0}" -f $env:RYZEN_AI_INSTALLATION_PATH) -ForegroundColor DarkGray
Write-Host ""

$staticModel = Join-Path $pkg "workloads\whisper\models\static-onnx"
$amdModel    = Join-Path $pkg "workloads\whisper\models\vendor\amd"
$audio       = Join-Path $pkg "workloads\audio\jfk.wav"

# (backend, device, model, cacheSubdir). -HotOnly skips CPU/GPU and reuses the NPU
# caches a prior full run compiled, so it's fast and free of the VitisAI compile spam.
$cases = if ($HotOnly) {
    @(
        @("auto", "npu", $staticModel, "sc-auto-npu"),
        @("amd",  "npu", $amdModel,    "sc-amd-npu")
    )
} else {
    @(
        @("onnx-static", "cpu", $staticModel, "sc-cpu"),
        @("onnx-static", "gpu", $staticModel, "sc-gpu"),
        @("auto",        "npu", $staticModel, "sc-auto-npu"),
        @("amd",         "npu", $amdModel,    "sc-amd-npu")
    )
}

foreach ($c in $cases) {
    $backend, $device, $model, $sub = $c
    $cache = Join-Path $pkg "sccache\$sub"
    Write-Host ("=== {0} / {1}{2} ===" -f $backend, $device, $(if ($HotOnly) { " (hot-only)" } else { "" })) -ForegroundColor Yellow
    $out = & powershell -NoProfile -Command {
        param($exe, $model, $audio, $backend, $device, $cache, $cleanPath, $hot)
        $env:Path = $cleanPath
        Remove-Item Env:\RYZEN_AI_INSTALLATION_PATH -ErrorAction SilentlyContinue
        $extra = @("--cache", $cache); if ($hot) { $extra += "--hot-only" }
        & $exe run whisper $model $audio $backend $device 3 @extra 2>&1
        Write-Output "___EXIT___$LASTEXITCODE"
    } -args $exe, $model, $audio, $backend, $device, $cache, $cleanPath, ([bool]$HotOnly)
    # Drop the VitisAI compiler's [DBG]/glog chatter so only the app's own report shows.
    $out | Where-Object { $_ -notmatch '^\s*(\[DBG\]|I\d{8}|WARNING: Logging|\(WARNING:)' } | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
}
