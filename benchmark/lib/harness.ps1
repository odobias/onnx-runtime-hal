# Backward-compatible barrel for benchmark scripts.
foreach ($module in @(
    'core.ps1', 'bootstrap.ps1', 'hardware.ps1', 'environment.ps1',
    'cache.ps1', 'execution.ps1', 'runner-pack.ps1', 'suite.ps1',
    'accuracy.ps1', 'suite-runner.ps1', 'executor-selection.ps1'
)) {
    . (Join-Path $PSScriptRoot $module)
}
