<#
.SYNOPSIS
    Compile and run src/tests/metrics_tests.cpp on its own.

.DESCRIPTION
    metrics.hpp is header-only with no dependencies, so its tests do not need
    the solution, ONNX Runtime or any vendor SDK. Building just this one
    translation unit turns a scorer change into a few seconds of feedback
    instead of a full runner build.
#>
[CmdletBinding()]
param(
    [string]$OutDir = "$PSScriptRoot\..\..\build\metrics-tests"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere))
{
    throw "vswhere not found at $vswhere"
}

$vs = (& $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath)
if ([string]::IsNullOrWhiteSpace($vs))
{
    throw "no Visual Studio installation with the x64 C++ toolset"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$exe = Join-Path $OutDir "metrics_tests.exe"

# cl needs the environment vcvars64 sets, and that is a batch file, so the only
# reliable way to get it is to run it and inherit what it exports.
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$source = Join-Path $root "src\tests\metrics_tests.cpp"
$include = Join-Path $root "src\runner\include"

# /utf-8 matters: the test file contains UTF-8 escapes and literals, and MSVC
# otherwise reads the source in the system code page.
#
# Objects land in the working directory rather than via /Fo, because a quoted
# path ending in a backslash -- which /Fo requires to mean "directory" -- has
# cmd treat the closing quote as escaped, and cl then reports a missing source
# file that is sitting right there on the command line.
$cmd = "`"$vcvars`" >nul && cl /nologo /std:c++20 /EHsc /W4 /WX /utf-8 " +
       "/DNPU_INFERENCE_BENCH_METRICS_TESTS_STANDALONE " +
       "/I`"$include`" /Fe:`"$exe`" `"$source`""

Write-Host "compiling metrics_tests.cpp" -ForegroundColor Cyan
Push-Location $OutDir
try
{
    & cmd /c $cmd
}
finally
{
    Pop-Location
}
if ($LASTEXITCODE -ne 0)
{
    throw "compilation failed with exit code $LASTEXITCODE"
}

Write-Host "running" -ForegroundColor Cyan
& $exe
$code = $LASTEXITCODE

if ($code -eq 0)
{
    # Then prove the Python port still agrees with what was just built, rather
    # than assuming it does because both have passing tests of their own.
    Write-Host "cross-checking the Python port" -ForegroundColor Cyan
    $cases = Join-Path $root "src\tests\normalization-cases.txt"
    $dump = Join-Path $OutDir "normalization-dump.tsv"

    # cmd redirection would write the system code page; .NET writes UTF-8 with no
    # BOM, which is what the Python side reads.
    $lines = & $exe --dump-corpus $cases
    if ($LASTEXITCODE -ne 0)
    {
        throw "corpus dump failed with exit code $LASTEXITCODE"
    }
    [IO.File]::WriteAllLines($dump, $lines, [Text.UTF8Encoding]::new($false))

    $python = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    & $python (Join-Path $root "tools\validate\scoring.py") --cross-check $dump
    $code = $LASTEXITCODE
}

Write-Host ("METRICS_TESTS_EXIT={0}" -f $code) -ForegroundColor $(if ($code -eq 0) { "Green" } else { "Red" })
exit $code
