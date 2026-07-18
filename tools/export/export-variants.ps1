# Exports a Whisper model to OpenVINO IR. The product ships a single portable static ONNX
# model across all platforms, so only the FP32 OV-IR baseline is a product variant; the
# quantized precisions (fp16/int8/int4) are kept for RESEARCH ONLY. To avoid the quant
# models leaking into the product benchmark, non-fp32 entries are written to a separate
# research manifest (models/manifest.research.json) and never to the product manifest.json.
#
#   .\export-variants.ps1                          # fp32 OV-IR baseline -> manifest.json
#   .\export-variants.ps1 -Formats fp16,int8,int4  # research quant sweep -> manifest.research.json
#   .\export-variants.ps1 -Model openai/whisper-tiny -Prefix wt

[CmdletBinding()]
param(
    [string]$Model = "openai/whisper-tiny.en",
    [string]$Prefix = "wten",
    [ValidateSet("fp32", "fp16", "int8", "int4")]
    [string[]]$Formats = @("fp32"),
    [string[]]$Devices = @("NPU", "GPU", "CPU")
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$models = Join-Path $root "models"
$variants = Join-Path $models "whisper\variants-ov"
New-Item -ItemType Directory -Force -Path $variants | Out-Null
$manifestPath = Join-Path $models "manifest.json"

# Resolve venv python (shared with get-model.ps1).
$venv = Join-Path $root "artifacts/venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $py) { Write-Host "Python not found. Run bootstrap.ps1 first." -ForegroundColor Red; exit 1 }
    Write-Host "Creating venv at $venv" -ForegroundColor Cyan
    & $py -m venv $venv
}
Write-Host "Ensuring export toolchain..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
# nncf is only needed for the research quant precisions; install it only when requested.
$needNncf = @($Formats | Where-Object { $_ -ne "fp32" }).Count -gt 0
$pkgs = @("optimum-intel[openvino]", "openvino-tokenizers")
if ($needNncf) { $pkgs += "nncf" }
& $vpy -m pip install --quiet @pkgs
$optimum = Join-Path $venv "Scripts\optimum-cli.exe"

# Human-readable description of what each weight-format actually does.
$methodOf = @{
    fp32 = "FP32 baseline (no compression)"
    fp16 = "FP16 weights (optimum-intel) [research only]"
    int8 = "INT8 weight-only compression (NNCF) [research only]"
    int4 = "INT4 weight-only compression (NNCF) [research only]"
}

$entries = @()
foreach ($fmt in $Formats) {
    $id = "$Prefix-ov-$fmt"
    $outDir = Join-Path $variants $fmt
    if (Test-Path (Join-Path $outDir "openvino_encoder_model.xml")) {
        Write-Host "[$fmt] already exported -> $outDir" -ForegroundColor Green
    }
    else {
        Write-Host "[$fmt] exporting -> $outDir" -ForegroundColor Cyan
        & $optimum export openvino --model $Model --weight-format $fmt `
            --task automatic-speech-recognition-with-past $outDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[$fmt] export FAILED; skipping in manifest." -ForegroundColor Yellow
            if ((& $vpy --version) -match "3\.1[4-9]") {
                Write-Host ("  Hint: on Python 3.14+, huggingface/optimum has a known bug (`"NormalizedConfig." + `
                        "__init__() got multiple values for argument 'allow_new'`", huggingface/optimum#2409, " + `
                        "abandoned/never merged). Workaround: in artifacts/venv\Lib\site-packages\optimum\exporters\base.py, " + `
                        "change `"self.NORMALIZED_CONFIG_CLASS(self._config)`" to " + `
                        "`"self.__class__.NORMALIZED_CONFIG_CLASS(self._config)`". Or use Python <=3.13.") -ForegroundColor Yellow
            }
            continue
        }
    }

    $sizeMb = [math]::Round(((Get-ChildItem $outDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
    $entries += [ordered]@{
        id         = $id
        backend    = "intel"
        precision  = $fmt
        method     = $methodOf[$fmt]
        model_dir  = "artifacts/workloads/whisper/models/ov-ir/$fmt"   # repo-relative for portability
        devices    = $Devices
        size_mb    = $sizeMb
    }
}

# Split entries: fp32 is a product variant (manifest.json); quantized precisions are
# research-only and go to manifest.research.json so they never enter the product benchmark.
$researchPath = Join-Path $models "manifest.research.json"

function Write-Manifest($path, $newEntries, $noteText) {
    if ($newEntries.Count -eq 0) { return }
    $existing = $null
    $preserved = @()
    if (Test-Path $path) {
        $existing = Get-Content $path -Raw | ConvertFrom-Json
        $newIds = @($newEntries | ForEach-Object { $_.id })
        $preserved = @($existing.variants | Where-Object { $newIds -notcontains $_.id })
    }
    $manifest = [ordered]@{
        model    = $(if ($existing -and $existing.model) { $existing.model } else { $Model })
        created  = (Get-Date).ToString("s")
        note     = $noteText
        variants = @($preserved + $newEntries)
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $path -Encoding UTF8
    Write-Host "`nWrote $path with $($newEntries.Count) variant(s):" -ForegroundColor Green
    $newEntries | ForEach-Object { Write-Host ("  {0,-14} {1,-5} {2} MB" -f $_.id, $_.precision, $_.size_mb) }
    if ($preserved.Count -gt 0) {
        Write-Host "Preserved $($preserved.Count) existing non-overwritten entr$(if ($preserved.Count -eq 1) { 'y' } else { 'ies' })." -ForegroundColor DarkGray
    }
}

$productEntries = @($entries | Where-Object { $_.precision -eq "fp32" })
$researchEntries = @($entries | Where-Object { $_.precision -ne "fp32" })

Write-Manifest $manifestPath $productEntries `
    "Backend-neutral variant manifest. Each platform appends entries with its own backend/model_dir/devices."
Write-Manifest $researchPath $researchEntries `
    "RESEARCH-ONLY quantized OV-IR variants. Not part of the product; not run by the default benchmark. Point benchmark-quant.ps1 -Manifest here to sweep them."

if ($researchEntries.Count -gt 0) {
    Write-Host "`nNote: quantized variants are research-only (manifest.research.json)." -ForegroundColor Yellow
    Write-Host "      Benchmark them explicitly: .\benchmark\research\sweep-quantization.ps1 -Manifest models\manifest.research.json" -ForegroundColor DarkGray
}
