#Requires -Version 7
<#
.SYNOPSIS
  Validate static-onnx-tiny-multi-7s on CPU / GPU (DirectML) / NPU.

.DESCRIPTION
  Fetches the HF package if missing, then runs the Python smoke for each
  requested device. For NPU on HAL suite machines, optionally also invokes
  benchmark\run-suite.ps1 against workload whisper-tiny-static-multi-7s.

  Other machines with an NPU EP should run:
    .\tools\validate\run_static_whisper_7s_multi.ps1 -Device npu
  and/or (after building npu_inference_bench):
    .\benchmark\run-suite.ps1 -Device npu -Only whisper -Provider auto

.PARAMETER Device
  One or more of: cpu, gpu, npu. Default: cpu,gpu (safe on non-NPU hosts).

.PARAMETER SkipFetch
  Do not attempt Hugging Face fetch when the local package is missing.

.PARAMETER Suite
  Also run the portable HAL suite for whisper on the same devices (needs build).

.PARAMETER Eval
  Also run multi-sample WER over eval.jsonl and the FLEURS multilingual clips
  (product 7s fit + full 30s canvas), each compared against the committed baseline
  in benchmark\baselines\whisper-static-multi-7s.json.

.PARAMETER UpdateBaseline
  Record this host's numbers as the baseline instead of comparing against it.
  Only for a trusted reference host.

.PARAMETER NoBaseline
  Run the evals without comparing against the baseline.

.PARAMETER PythonCpu
  Python for CPU smoke (default: .venv).

.PARAMETER PythonGpu
  Python with onnxruntime-directml for GPU (default: .venv-dml).
#>
param(
    [ValidateSet("cpu", "gpu", "npu")]
    [string[]]$Device = @("cpu", "gpu"),

    [switch]$SkipFetch,
    [switch]$Suite,
    [switch]$Eval,
    [switch]$UpdateBaseline,
    [switch]$NoBaseline,

    [string]$PythonCpu = "",
    [string]$PythonGpu = "",
    [string]$PythonNpu = ""
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch {}
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

$model = Join-Path $root "artifacts\workloads\whisper\models\static-onnx-tiny-multi-7s"
$audio = Join-Path $root "artifacts\workloads\speech\jfk.wav"
$smoke = Join-Path $root "tools\validate\smoke_static_whisper_7s_multi.py"
$evalPy = Join-Path $root "tools\validate\eval_static_whisper_7s_multi.py"
$multiPy = Join-Path $root "tools\validate\eval_static_whisper_multilingual.py"
$multiManifest = Join-Path $root "artifacts\workloads\speech\multilingual\manifest.jsonl"

function Resolve-Py([string]$explicit, [string[]]$candidates) {
    if ($explicit -and (Test-Path -LiteralPath $explicit)) { return (Resolve-Path $explicit).Path }
    foreach ($c in $candidates) {
        $p = Join-Path $root $c
        if (Test-Path -LiteralPath $p) { return (Resolve-Path $p).Path }
    }
    return $null
}

$pyCpu = Resolve-Py $PythonCpu @(".venv\Scripts\python.exe", ".venv-dml\Scripts\python.exe")
$pyGpu = Resolve-Py $PythonGpu @(".venv-dml\Scripts\python.exe", ".venv\Scripts\python.exe")
$pyNpu = Resolve-Py $PythonNpu @(".venv-dml\Scripts\python.exe", ".venv\Scripts\python.exe")

if (-not (Test-Path -LiteralPath $smoke)) {
    throw "Missing smoke script: $smoke"
}

if (-not (Test-Path -LiteralPath (Join-Path $model "encoder_model.onnx"))) {
    if ($SkipFetch) {
        throw "Model package missing at $model (SkipFetch set)."
    }
    Write-Host "==> Fetching static-onnx-tiny-multi-7s from HF..." -ForegroundColor Cyan
    $fetch = Join-Path $root "tools\fetch\get-models.ps1"
    if (-not (Test-Path -LiteralPath $fetch)) {
        throw "Model missing and fetch script not found: $fetch"
    }
    & $fetch
    if (-not (Test-Path -LiteralPath (Join-Path $model "encoder_model.onnx"))) {
        throw "Fetch completed but encoder still missing under $model"
    }
}

if (-not (Test-Path -LiteralPath $audio)) {
    throw "Missing audio fixture: $audio (run tools\fetch\get-models.ps1)"
}

$results = [System.Collections.Generic.List[object]]::new()
$failed = $false

foreach ($dev in $Device) {
    $py = switch ($dev) {
        "cpu" { $pyCpu }
        "gpu" { $pyGpu }
        "npu" { $pyNpu }
    }
    if (-not $py) {
        Write-Host "FAIL [$dev]: no Python interpreter found" -ForegroundColor Red
        $results.Add([pscustomobject]@{ Device = $dev; Pass = $false; Detail = "no python" })
        $failed = $true
        continue
    }

    Write-Host ""
    Write-Host "==> Smoke device=$dev python=$py" -ForegroundColor Cyan
    & $py $smoke --device $dev
    $code = $LASTEXITCODE
    $pass = ($code -eq 0)
    if (-not $pass) { $failed = $true }
    $results.Add([pscustomobject]@{
            Device = $dev
            Pass   = $pass
            Exit   = $code
            Python = $py
        })
    Write-Host ("[{0}] exit={1}" -f $dev, $code) -ForegroundColor $(if ($pass) { "Green" } else { "Red" })
}

if ($Eval) {
    if (-not (Test-Path -LiteralPath $evalPy)) {
        throw "Missing eval script: $evalPy"
    }
    # Every eval prints its WER delta against benchmark/baselines/whisper-static-multi-7s.json
    # unless the caller is (re)recording that baseline on a reference host.
    $baselineArgs = @()
    if ($UpdateBaseline) { $baselineArgs += "--update-baseline" }
    elseif ($NoBaseline) { $baselineArgs += "--no-baseline" }
    foreach ($dev in $Device) {
        $py = switch ($dev) {
            "cpu" { $pyCpu }
            "gpu" { $pyGpu }
            "npu" { $pyNpu }
        }
        if (-not $py) {
            Write-Host "FAIL eval [$dev]: no Python" -ForegroundColor Red
            $failed = $true
            continue
        }
        foreach ($winArgs in @(
                @("--window", "product", "--only-fit"),
                @("--window", "full")
            )) {
            Write-Host ""
            Write-Host ("==> Eval device={0} {1}" -f $dev, ($winArgs -join ' ')) -ForegroundColor Cyan
            & $py $evalPy --device $dev @winArgs $baselineArgs
            if ($LASTEXITCODE -ne 0) {
                $failed = $true
                Write-Host "FAIL eval device=$dev exit=$LASTEXITCODE" -ForegroundColor Red
            }
        }

        if (-not (Test-Path -LiteralPath $multiManifest)) {
            Write-Host "SKIP multilingual eval [$dev]: $multiManifest missing (run tools\fetch\get-models.ps1)" -ForegroundColor DarkYellow
            continue
        }
        foreach ($win in @("product", "full")) {
            Write-Host ""
            Write-Host ("==> Multilingual eval device={0} --window {1}" -f $dev, $win) -ForegroundColor Cyan
            & $py $multiPy --device $dev --window $win --lang-mode auto $baselineArgs
            if ($LASTEXITCODE -ne 0) {
                $failed = $true
                Write-Host "FAIL multilingual eval device=$dev window=$win exit=$LASTEXITCODE" -ForegroundColor Red
            }
        }
    }
}

if ($Suite) {
    $suite = Join-Path $root "benchmark\run-suite.ps1"
    if (-not (Test-Path -LiteralPath $suite)) {
        throw "Suite script missing: $suite"
    }
    foreach ($dev in $Device) {
        Write-Host ""
        Write-Host "==> HAL suite -Device $dev -Only whisper (includes whisper-tiny-static-multi-7s)" -ForegroundColor Cyan
        & $suite -Device $dev -Only whisper -Provider auto
        if ($LASTEXITCODE -ne 0) {
            $failed = $true
            Write-Host "FAIL suite device=$dev exit=$LASTEXITCODE" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
$results | Format-Table -AutoSize | Out-String | Write-Host

$reportDir = Join-Path $root "results\reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
$lines = @(
    "# Smoke run: static-onnx-tiny-multi-7s",
    "",
    "Architecture: $env:PROCESSOR_ARCHITECTURE",
    "When: $stamp",
    "Devices: $($Device -join ', ')",
    "",
    "| Device | Pass | Exit | Python |",
    "|--------|------|------|--------|"
)
foreach ($r in $results) {
    $lines += "| $($r.Device) | $($r.Pass) | $($r.Exit) | ``$($r.Python)`` |"
}
$out = Join-Path $reportDir "whisper-static-multi-7s-smoke-latest.md"
$lines -join "`n" | Set-Content -LiteralPath $out -Encoding utf8
Write-Host "Wrote $out"

if ($failed) { exit 1 }
exit 0
