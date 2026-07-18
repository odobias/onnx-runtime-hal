# Canonical cache addressing for all benchmark entry points.
function Get-BenchmarkCachePath {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Platform,
        [Parameter(Mandatory)][string]$Provider,
        [Parameter(Mandatory)][string]$Workload,
        [Parameter(Mandatory)][string]$Profile,
        [Parameter(Mandatory)][string]$Device
    )
    $segments = @($Platform, $Provider, $Workload, $Profile, $Device) |
        ForEach-Object { ([string]$_ -replace '[^A-Za-z0-9_.-]', '_') }
    return Join-Path $Root ("artifacts\cache\benchmark\" + ($segments -join '\'))
}

function Reset-BenchmarkCache {
    param([Parameter(Mandatory)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
    Write-Host "cache      : reset -> $Path" -ForegroundColor DarkGray
    return $Path
}
