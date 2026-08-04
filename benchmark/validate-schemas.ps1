<#
.SYNOPSIS
Fails when benchmark CSV schemas drift between their contracts and writers.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "lib\harness.ps1")

function Get-JsonColumns {
    param([Parameter(Mandatory)][string]$Path)
    return @((Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json).columns)
}

function Get-CppHeaderColumns {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Constant
    )

    $text = Get-Content -LiteralPath $Source -Raw -Encoding UTF8
    $pattern = 'constexpr\s+const\s+char\*\s+' + [regex]::Escape($Constant) +
        '\s*=\s*(?<body>(?:\s*"[^"]*"\s*)+);'
    $definition = [regex]::Match($text, $pattern)
    if (-not $definition.Success) {
        throw "Could not parse C++ CSV header constant '$Constant' in $Source"
    }
    $parts = [regex]::Matches($definition.Groups["body"].Value, '"(?<value>[^"]*)"')
    return @((($parts | ForEach-Object { $_.Groups["value"].Value }) -join "").Split(","))
}

function Assert-ColumnsEqual {
    param(
        [Parameter(Mandatory)][string]$ExpectedName,
        [Parameter(Mandatory)][string[]]$Expected,
        [Parameter(Mandatory)][string]$ActualName,
        [Parameter(Mandatory)][string[]]$Actual
    )

    $duplicates = @($Actual | Group-Object | Where-Object Count -gt 1)
    if ($duplicates.Count) {
        throw "$ActualName contains duplicate columns: $(($duplicates.Name) -join ', ')"
    }
    if ($Expected.Count -ne $Actual.Count) {
        throw "$ActualName has $($Actual.Count) columns; $ExpectedName has $($Expected.Count)"
    }
    for ($i = 0; $i -lt $Expected.Count; ++$i) {
        if ($Expected[$i] -cne $Actual[$i]) {
            throw "$ActualName differs from $ExpectedName at column $i`: expected '$($Expected[$i])', got '$($Actual[$i])'"
        }
    }
}

$asr = Get-JsonColumns (Join-Path $root "benchmark\schemas\asr-results.columns.json")
$classifier = Get-JsonColumns (Join-Path $root "benchmark\schemas\classifier-results.columns.json")
$cppSource = Join-Path $root "src\runner\benchmark\benchmark_ledger.ipp"

Assert-ColumnsEqual "ASR JSON schema" $asr "C++ ASR writer" `
    (Get-CppHeaderColumns $cppSource "kBenchmarkCsvHeader")
Assert-ColumnsEqual "ASR JSON schema" $asr "PowerShell ASR writer" `
    @(Get-BenchmarkResultColumns)
Assert-ColumnsEqual "classifier JSON schema" $classifier "C++ classifier writer" `
    (Get-CppHeaderColumns $cppSource "kClassifierCsvHeader")

Write-Host "Benchmark CSV schemas are consistent." -ForegroundColor Green
