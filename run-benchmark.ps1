# Package entry point. All normal arguments are forwarded unchanged to the
# established benchmark/run-suite.ps1 interface.
$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$manifest = Join-Path $PSScriptRoot "runner-package.json"
$forward = [System.Collections.Generic.List[string]]::new()
$list = $false
$explain = $false
foreach ($argument in $args) {
    if ([string]$argument -eq "--list-runners") {
        $list = $true
    } elseif ([string]$argument -eq "--explain") {
        $explain = $true
    } else {
        $forward.Add([string]$argument)
    }
}

if ($list) {
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "This directory has no runner-package.json. Build a package first."
    }
    $package = Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host ("Package: {0} ({1})" -f $package.package_id, $package.architecture)
    foreach ($runner in @($package.runners | Sort-Object id)) {
        Write-Host ("  {0,-10} runtime={1,-7} vendors={2} path={3}" -f
            $runner.id, $runner.runtime_target, (@($runner.vendors) -join ","), $runner.path)
    }
    foreach ($runner in @($package.skipped | Sort-Object id)) {
        Write-Host ("  {0,-10} skipped: {1}" -f $runner.id, $runner.reason)
    }
    exit 0
}

$suite = Join-Path $PSScriptRoot "benchmark\run-suite.ps1"
if (-not (Test-Path -LiteralPath $suite)) {
    throw "Benchmark suite script is missing: $suite"
}
if ($explain) { $forward.Add("-ExplainRunnerSelection") }
$forwardArguments = $forward.ToArray()
$shellExe = (Get-Process -Id $PID).Path
if (Test-Path -LiteralPath $manifest) {
    & $shellExe -NoLogo -NoProfile -File $suite `
        -RunnerManifest $manifest @forwardArguments
} else {
    & $shellExe -NoLogo -NoProfile -File $suite @forwardArguments
}
exit $LASTEXITCODE
