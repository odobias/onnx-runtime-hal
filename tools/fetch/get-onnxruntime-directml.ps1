[CmdletBinding()]
param(
    [string]$Version = "",
    [ValidateSet("x64", "arm64")]
    [string]$Architecture = $(if (
        [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq
        [System.Runtime.InteropServices.Architecture]::Arm64
    ) { "arm64" } else { "x64" })
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination = Join-Path $root "third_party\onnxruntime-directml"
$work = Join-Path $root "build\downloads\onnxruntime-directml"
$packageId = "microsoft.ml.onnxruntime.directml"
$indexUrl = "https://api.nuget.org/v3-flatcontainer/$packageId/index.json"

if (-not $Version) {
    $index = Invoke-RestMethod -Uri $indexUrl
    $stable = @($index.versions | Where-Object { $_ -notmatch "-" })
    if ($stable.Count -eq 0) {
        throw "NuGet returned no stable versions for $packageId"
    }
    $Version = $stable[-1]
}

$versionLower = $Version.ToLowerInvariant()
$packageUrl = "https://api.nuget.org/v3-flatcontainer/$packageId/$versionLower/$packageId.$versionLower.nupkg"
$nupkg = Join-Path $work "$packageId.$versionLower.nupkg"
$archive = Join-Path $work "$packageId.$versionLower.zip"
$expanded = Join-Path $work "expanded"
$directmlExpanded = Join-Path $work "directml-expanded"
$staging = Join-Path $work "staging"

New-Item -ItemType Directory -Force -Path $work | Out-Null
Write-Host "Downloading ONNX Runtime DirectML $Version ($Architecture)..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $packageUrl -OutFile $nupkg
Copy-Item $nupkg $archive -Force

Remove-Item $expanded, $directmlExpanded, $staging -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $archive -DestinationPath $expanded -Force

$includeSource = Join-Path $expanded "build\native\include"
$nativeSource = Join-Path $expanded "runtimes\win-$Architecture\native"
foreach ($required in @(
    (Join-Path $includeSource "onnxruntime_cxx_api.h"),
    (Join-Path $nativeSource "onnxruntime.lib"),
    (Join-Path $nativeSource "onnxruntime.dll")
)) {
    if (-not (Test-Path $required)) {
        throw "DirectML NuGet package is missing expected file: $required"
    }
}

# The ORT package declares Microsoft.AI.DirectML as a NuGet dependency rather
# than embedding DirectML.dll. Resolve that exact dependency version from the
# package metadata so the two runtimes cannot silently drift.
[xml]$nuspec = Get-Content (Join-Path $expanded "Microsoft.ML.OnnxRuntime.DirectML.nuspec") -Raw
$namespace = [System.Xml.XmlNamespaceManager]::new($nuspec.NameTable)
$namespace.AddNamespace("n", $nuspec.DocumentElement.NamespaceURI)
$directmlDependency = $nuspec.SelectSingleNode(
    "//n:dependency[@id='Microsoft.AI.DirectML']",
    $namespace
)
if (-not $directmlDependency) {
    throw "ORT DirectML package does not declare Microsoft.AI.DirectML"
}
$directmlVersion = $directmlDependency.version
$directmlId = "microsoft.ai.directml"
$directmlPackage = Join-Path $work "$directmlId.$directmlVersion.nupkg"
$directmlArchive = Join-Path $work "$directmlId.$directmlVersion.zip"
$directmlUrl = "https://api.nuget.org/v3-flatcontainer/$directmlId/$directmlVersion/$directmlId.$directmlVersion.nupkg"
Invoke-WebRequest -Uri $directmlUrl -OutFile $directmlPackage
Copy-Item $directmlPackage $directmlArchive -Force
Expand-Archive -Path $directmlArchive -DestinationPath $directmlExpanded -Force

$architecturePattern = if ($Architecture -eq "arm64") { "arm64" } else { "x64" }
$directmlDll = @(
    Get-ChildItem $directmlExpanded -Recurse -Filter "DirectML.dll" |
        Where-Object { $_.FullName.ToLowerInvariant().Contains($architecturePattern) }
) | Select-Object -First 1
if (-not $directmlDll) {
    throw "Microsoft.AI.DirectML $directmlVersion has no DirectML.dll for $Architecture"
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $staging "include"), `
    (Join-Path $staging "lib"), `
    (Join-Path $staging "bin") | Out-Null
Copy-Item (Join-Path $includeSource "*") (Join-Path $staging "include") -Recurse -Force
Copy-Item (Join-Path $nativeSource "*.lib") (Join-Path $staging "lib") -Force
Copy-Item (Join-Path $nativeSource "*.dll") (Join-Path $staging "bin") -Force
Copy-Item $directmlDll.FullName (Join-Path $staging "bin\DirectML.dll") -Force
[System.IO.File]::WriteAllText(
    (Join-Path $staging "VERSION"),
    "$Version`n",
    $utf8
)

Remove-Item $destination -Recurse -Force -ErrorAction SilentlyContinue
Move-Item $staging $destination

Write-Host "Installed: $destination" -ForegroundColor Green
Write-Host "Build with: .\tools\build\build.ps1 -EnableOrt -DisableIntel" -ForegroundColor Cyan
