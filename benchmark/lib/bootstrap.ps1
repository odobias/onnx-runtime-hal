# Benchmark library module. Dot-source benchmark/lib/harness.ps1 instead of loading this directly.

function Get-BenchmarkOvepExe([string]$Root, [string]$Configuration = "Release") {
    return (Join-Path $Root "build\x64-ovep\$Configuration\NpuInferenceBench.exe")
}

# Choose the exe to run a variant with. On an Intel host the neutral self-selecting
# variant ("auto"/onnx-static/ort backends) runs through the unified OVEP binary --
# the default Intel build is OpenVINO GenAI and has no ONNX Runtime backend compiled
# in, so it cannot serve the neutral static-ONNX path. Everything else (the OV-IR
# variants, and every non-Intel host) uses the default exe.
function Get-BenchmarkExeForVariant($Variant, [string]$DefaultExe, [string]$Root, [string]$HostVendor, [string]$Configuration = "Release") {
    if ($HostVendor -eq 'Intel' -and (Get-BenchmarkBackendVendor $Variant.backend) -eq 'Neutral') {
        $ovep = Get-BenchmarkOvepExe $Root $Configuration
        if (Test-Path $ovep) { return $ovep }
    }
    return $DefaultExe
}

# --- environment bootstrap ---------------------------------------------------

# Invoke a sibling setup/build script in an ISOLATED child PowerShell process.
# This is deliberate: get-eval-set.ps1 ends in `exit 0`, build.ps1/get-models.ps1
# `exit 1` on failure, and in PowerShell an `exit` inside a `& .\child.ps1` call
# terminates the PARENT script too -- which would silently kill the benchmark
# before the sweep. A child process contains the exit code so we can react to it.
function Invoke-BenchmarkChildScript {
    param(
        [Parameter(Mandatory)] [string]$ScriptPath,
        [string[]]$ScriptArgs = @(),
        [switch]$Fatal
    )
    $psExe = $null
    try { $psExe = (Get-Process -Id $PID -ErrorAction Stop).Path } catch { }
    if (-not $psExe) { $psExe = "powershell" }
    & $psExe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @ScriptArgs
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0 -and $Fatal) {
        throw ("{0} failed (exit {1})" -f [System.IO.Path]::GetFileName($ScriptPath), $code)
    }
    return $code
}

# Merge (idempotently) the neutral unified `auto` variant into a generated manifest.
# The unified binary self-selects its execution provider (NPU>GPU>CPU) at runtime, so
# this one entry runs the same self-selecting benchmark on any host. Runs on the
# portable static ONNX model. Returns $true if the manifest was written. No-op (returns
# $false) when the manifest or model is absent -- there is nothing to attach the entry to.
function Add-BenchmarkUnifiedAutoVariant {
    param(
        [Parameter(Mandatory)] [string]$Manifest,
        [Parameter(Mandatory)] [string]$Models
    )
    $modelRel = "src/workloads/whisper/models/static-onnx"
    $modelDir = Join-Path $Models "whisper\en-static-onnx"
    if (-not (Test-Path $Manifest) -or -not (Test-Path $modelDir)) { return $false }

    $sizeMb = [math]::Round(((Get-ChildItem $modelDir -Recurse -File -ErrorAction SilentlyContinue |
                Measure-Object Length -Sum).Sum / 1MB), 1)
    $entry = [ordered]@{
        id        = "whisper-tiny-unified-auto"
        backend   = "auto"
        precision = "fp32-static"
        method    = "unified ONNX Runtime; binary self-selects the EP (NPU>GPU>CPU)"
        model_dir = $modelRel
        devices   = @("auto")
        size_mb   = $sizeMb
    }
    $manifestObj = Get-Content $Manifest -Raw | ConvertFrom-Json
    $variants = @($manifestObj.variants | Where-Object { $_.id -ne $entry.id })
    $manifestObj.variants = @($variants + $entry)
    $manifestObj | ConvertTo-Json -Depth 8 | Set-Content -Path $Manifest -Encoding UTF8
    return $true
}

# Ensure everything a sweep needs exists, building the app and fetching
# src/workloads/eval/audio on demand so `benchmark-quant.ps1` works from a fresh checkout.
# Idempotent: every step is skipped when its output is already present. This
# assumes the toolchain + vendor SDK are installed (that is bootstrap.ps1's job);
# if the build can't run it says exactly that. Model fetches for the neutral
# onnx-static artifact require an authenticated `hf` CLI on PATH and are
# best-effort -- a missing private snapshot won't abort the run.
function Initialize-BenchmarkEnvironment {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [Parameter(Mandatory)] [string]$Exe,
        [Parameter(Mandatory)] [string]$Manifest,
        [Parameter(Mandatory)] [string]$EvalSet,
        [string]$Configuration = "Release",
        [object]$Platform = $null
    )
    if (-not $Platform) { $Platform = Get-BenchmarkPlatform }
    $vendor  = $Platform.npu_vendor
    $scripts = Join-Path $Root "scripts"
    $models  = Join-Path $Root "models"

    Write-Host ("== Bootstrap: preparing benchmark prerequisites (host {0}) ==" -f $vendor) -ForegroundColor Cyan

    # 1) Build the app for this host's backends if it is not built yet.
    if (-not (Test-Path $Exe)) {
        $buildArgs = @("-Configuration", $Configuration)
        switch ($vendor) {
            'AMD'      { $buildArgs += @("-EnableAmd", "-DisableIntel") }
            'Qualcomm' { $buildArgs += @("-EnableQualcomm", "-DisableIntel") }
            'Intel'    { }                       # Intel/OpenVINO is the default build
            default    { $buildArgs += @("-EnableOrt") }  # unknown host: at least the neutral ORT backend
        }
        Write-Host ("  - build app ({0})" -f ($buildArgs -join ' ')) -ForegroundColor DarkCyan
        Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "build.ps1") -ScriptArgs $buildArgs -Fatal | Out-Null
        if (-not (Test-Path $Exe)) {
            throw ("app build did not produce {0}. Install the toolchain/SDK first: .\tools\build\bootstrap.ps1 -Platform {1}" -f $Exe, $vendor.ToLower())
        }
    } else {
        Write-Host "  - app already built" -ForegroundColor DarkGray
    }

    # 1b) Intel native path: also build the unified OVEP binary (ONNX Runtime +
    #     OpenVINO EP) so the self-selecting `auto` variant runs on the Intel
    #     NPU/GPU/CPU through ORT, mirroring AMD (VitisAI) and Qualcomm (QNN).
    #     Best-effort -- a failure here leaves the default Intel (GenAI) variants
    #     fully working; only the neutral ORT path is skipped.
    if ($vendor -eq 'Intel') {
        $ovepExe = Get-BenchmarkOvepExe $Root $Configuration
        if (-not (Test-Path $ovepExe)) {
            try {
                $ortLib = Join-Path $Root "third_party\onnxruntime-openvino\lib\onnxruntime.lib"
                if (-not (Test-Path $ortLib)) {
                    Write-Host "  - assemble ORT + OpenVINO EP distro (setup-ovep.ps1)" -ForegroundColor DarkCyan
                    Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "setup-ovep.ps1") | Out-Null
                }
                if (Test-Path $ortLib) {
                    Write-Host "  - build unified OVEP binary (build.ps1 -EnableOvep)" -ForegroundColor DarkCyan
                    Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "build.ps1") `
                        -ScriptArgs @("-Configuration", $Configuration, "-EnableOvep") | Out-Null
                }
            } catch {
                Write-Host ("  ! OVEP setup failed ({0}); Intel native ORT path will be skipped." -f $_.Exception.Message) -ForegroundColor Yellow
            }
            if (-not (Test-Path $ovepExe)) {
                Write-Host "  ! unified OVEP binary unavailable; the 'auto' variant will use the default exe on Intel." -ForegroundColor Yellow
            }
        } else {
            Write-Host "  - unified OVEP binary already built" -ForegroundColor DarkGray
        }
    }

    # 2) Sample audio (public) -- also the fallback clip for the eval set.
    if (-not (Test-Path (Join-Path $models "audio\jfk.wav"))) {
        Write-Host "  - fetch sample audio" -ForegroundColor DarkCyan
        Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-audio.ps1") | Out-Null
    }

    # 3) Models for this host's runnable variants. The private snapshot
    #    (get-models.ps1) carries the neutral onnx-static model + base manifest;
    #    the public per-vendor scripts fill in vendor models and merge their entry.
    $staticEnc = Join-Path $models "whisper\en-static-onnx\encoder_model.onnx"
    if (-not (Test-Path $Manifest) -or -not (Test-Path $staticEnc)) {
        if (Get-Command hf -ErrorAction SilentlyContinue) {
            Write-Host "  - fetch model snapshot (private HF; best-effort)" -ForegroundColor DarkCyan
            Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-models.ps1") | Out-Null
        } else {
            Write-Host "  ! neutral onnx-static model missing and 'hf' CLI not on PATH; skipping private snapshot." -ForegroundColor Yellow
        }
    }
    switch ($vendor) {
        'AMD' {
            if (-not (Test-Path (Join-Path $models "whisper\amd\tiny_encoder.onnx")) -or -not (Test-Path $Manifest)) {
                Write-Host "  - fetch AMD model + merge manifest" -ForegroundColor DarkCyan
                Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-amd-model.ps1") | Out-Null
            }
        }
        'Intel' {
            if (-not (Test-Path $Manifest)) {
                Write-Host "  - export Intel OpenVINO model" -ForegroundColor DarkCyan
                Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-model.ps1") | Out-Null
            }
        }
    }

    # 4) Eval set (public LibriSpeech, or jfk fallback).
    if (-not (Test-Path $EvalSet)) {
        Write-Host "  - build eval set" -ForegroundColor DarkCyan
        Invoke-BenchmarkChildScript -ScriptPath (Join-Path $scripts "get-eval-set.ps1") | Out-Null
    }

    # 5) Register the neutral self-selecting variant. The manifest is generated
    #    (gitignored), so this tracked step re-injects the unified `auto` entry on
    #    every run whenever the portable static ONNX model is present -- letting the
    #    same self-selecting benchmark run on any platform (x64 or ARM64).
    if (Test-Path $staticEnc) {
        if (Add-BenchmarkUnifiedAutoVariant -Manifest $Manifest -Models $models) {
            Write-Host "  - registered unified 'auto' variant (self-selecting EP)" -ForegroundColor DarkCyan
        }
    }

    # Final gate: these three are non-negotiable for a sweep.
    $missing = @($Manifest, $EvalSet, $Exe | Where-Object { -not (Test-Path $_) })
    if ($missing.Count) {
        throw ("bootstrap could not produce required prerequisites:`n  {0}`nRun .\tools\build\bootstrap.ps1 -Platform {1} to install the toolchain/SDK and models." -f `
                ($missing -join "`n  "), $vendor.ToLower())
    }
    Write-Host "== Bootstrap: ready ==" -ForegroundColor Green
}

# --- detailed hardware inventory ---------------------------------------------

# Heuristic PCI (vendor:device) -> NPU architecture map. Windows reports only a
# generic device name ("NPU Compute Accelerator Device" on AMD), so the architecture
# is DERIVED from the PCI hardware id, not read from the device. Always cross-check
# against the raw pci_id the descriptor also carries; unknown ids return blank.
