# Downloads the AMD RyzenAI Whisper Tiny ONNX model plus tokenizer/preprocessor
# sidecars into one HAL model directory.
#
#   .\scripts\get-amd-model.ps1
#   .\scripts\get-amd-model.ps1 -Out whisper-tiny-amd

[CmdletBinding()]
param(
    [string]$Out = "whisper-tiny-amd"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = Split-Path $PSScriptRoot -Parent
$models = Join-Path $root "models"
$outDir = Join-Path $models $Out
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Get-File {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$OutFile
    )
    if (Test-Path $OutFile) {
        Write-Host "Already present: $OutFile" -ForegroundColor DarkGray
        return
    }
    Write-Host "Downloading $Uri" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile
}

$amdBase = "https://huggingface.co/amd/whisper-tiny-onnx-npu/resolve/main"
Get-File "$amdBase/tiny_encoder.onnx?download=true" (Join-Path $outDir "tiny_encoder.onnx")
Get-File "$amdBase/tiny_decoder.onnx?download=true" (Join-Path $outDir "tiny_decoder.onnx")

$openaiBase = "https://huggingface.co/openai/whisper-tiny/resolve/main"
$tokenizerFiles = @(
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json"
)
foreach ($file in $tokenizerFiles) {
    Get-File "$openaiBase/${file}?download=true" (Join-Path $outDir $file)
}

@'
{
    "passes": [
        {
            "name": "init",
            "plugin": "vaip-pass_init"
        },
        {
            "name": "vaiml_partition",
            "plugin": "vaip-pass_vaiml_partition",
            "vaiml_config": {
                "optimize_level": 3,
                "fe_experiment": "use-accurate-mode=LayerNorm2PassAdf",
                "aiecompiler_args": "--system-stack-size=512"
            }
        }
    ],
    "target": "VAIML",
    "targets": [
        {
            "name": "VAIML",
            "pass": [
                "init",
                "vaiml_partition"
            ]
        }
    ]
}
'@ | Set-Content -Encoding ASCII (Join-Path $outDir "vitisai_config_whisper_encoder.json")

@'
{
    "passes": [
        {
            "name": "init",
            "plugin": "vaip-pass_init"
        },
        {
            "name": "vaiml_partition",
            "plugin": "vaip-pass_vaiml_partition",
            "vaiml_config": {
                "optimize_level": 3,
                "aiecompiler_args": "--system-stack-size=512"
            }
        }
    ],
    "target": "VAIML",
    "targets": [
        {
            "name": "VAIML",
            "pass": [
                "init",
                "vaiml_partition"
            ]
        }
    ]
}
'@ | Set-Content -Encoding ASCII (Join-Path $outDir "vitisai_config_whisper_decoder.json")

Write-Host "AMD model ready: $outDir" -ForegroundColor Green
