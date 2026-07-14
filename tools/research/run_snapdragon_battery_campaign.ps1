[CmdletBinding()]
param([int]$Repeats = 3)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$OutputEncoding = [Text.UTF8Encoding]::new()

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class BenchmarkExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$suite = Join-Path $root "benchmark\run-suite.ps1"

try {
    [void][BenchmarkExecutionState]::SetThreadExecutionState($continuous -bor $systemRequired)
    for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
        Write-Host "`n===== Snapdragon battery latency pass $repeat / $Repeats =====" -ForegroundColor Magenta
        & $suite -Mode latency -Runtime all -Device npu,gpu,cpu -Only all `
            -Runs 5 -ClassifierRuns 20
        if ($LASTEXITCODE -ne 0) { throw "Latency pass $repeat failed with exit $LASTEXITCODE." }
    }

    Write-Host "`n===== Whole-graph FakeAudio NPU diagnostic latency =====" -ForegroundColor Magenta
    & $suite -Mode latency -Runtime all -Device npu -Only fakeaudio `
        -Profile npu-whole-diagnostic -Runs 5 -ClassifierRuns 20
    if ($LASTEXITCODE -ne 0) { throw "Diagnostic latency run failed with exit $LASTEXITCODE." }

    Write-Host "`n===== Whole-graph FakeAudio NPU diagnostic accuracy =====" -ForegroundColor Magenta
    & $suite -Mode accuracy-quick -Runtime all -Device npu -Only fakeaudio `
        -Profile npu-whole-diagnostic `
        -AccuracyLedger (Join-Path $root "results\ledgers\accuracy-diagnostics.jsonl")
    if ($LASTEXITCODE -ne 0) { throw "Diagnostic accuracy run failed with exit $LASTEXITCODE." }
} finally {
    [void][BenchmarkExecutionState]::SetThreadExecutionState($continuous)
}
