# Exports a Whisper model to several OpenVINO quantization variants and writes a
# backend-neutral manifest.json describing each. The manifest is what the benchmark
# harness consumes; adding a non-Intel backend later means appending entries with a
# different "backend" (e.g. "amd"/"qualcomm") and their own model_dir/devices --
# no harness changes required.
#
#   .\export-variants.ps1                          # fp32,fp16,int8,int4 of whisper-tiny.en
#   .\export-variants.ps1 -Formats fp16,int8       # subset
#   .\export-variants.ps1 -Model openai/whisper-tiny -Prefix wt

[CmdletBinding()]
param(
    [string]$Model = "openai/whisper-tiny.en",
    [string]$Prefix = "wten",
    [ValidateSet("fp32", "fp16", "int8", "int4")]
    [string[]]$Formats = @("fp32", "fp16", "int8", "int4"),
    [string[]]$Devices = @("NPU", "GPU", "CPU")
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$root = Split-Path $PSScriptRoot -Parent
$models = Join-Path $root "models"
$variants = Join-Path $models "variants"
New-Item -ItemType Directory -Force -Path $variants | Out-Null

# Resolve venv python (shared with get-model.ps1).
$venv = Join-Path $root ".venv"
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
& $vpy -m pip install --quiet "optimum-intel[openvino]" openvino-tokenizers nncf
$optimum = Join-Path $venv "Scripts\optimum-cli.exe"

# Human-readable description of what each weight-format actually does.
$methodOf = @{
    fp32 = "FP32 baseline (no compression)"
    fp16 = "FP16 weights (optimum-intel)"
    int8 = "INT8 weight-only compression (NNCF)"
    int4 = "INT4 weight-only compression (NNCF)"
}

$entries = @()
foreach ($fmt in $Formats) {
    $id = "$Prefix-ov-$fmt"
    $outDir = Join-Path $variants $id
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
                        "abandoned/never merged). Workaround: in .venv\Lib\site-packages\optimum\exporters\base.py, " + `
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
        model_dir  = "models/variants/$id"   # repo-relative for portability
        devices    = $Devices
        size_mb    = $sizeMb
    }
}

$manifest = [ordered]@{
    model    = $Model
    created  = (Get-Date).ToString("s")
    note     = "Backend-neutral variant manifest. Add AMD/Qualcomm variants by appending entries with backend=amd|qualcomm."
    variants = $entries
}

$manifestPath = Join-Path $models "manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host "`nWrote $manifestPath with $($entries.Count) variant(s):" -ForegroundColor Green
$entries | ForEach-Object { Write-Host ("  {0,-14} {1,-5} {2} MB" -f $_.id, $_.precision, $_.size_mb) }
